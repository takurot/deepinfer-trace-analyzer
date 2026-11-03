from decimal import Decimal as D

import pytest

from builders import TraceBuilder
from deepinfer_trace_analyzer.errors import AnalysisError
from deepinfer_trace_analyzer.summary import Options, summarize


def trace_file(tmp_path, builder):
    return builder.write(tmp_path / "t.json")


def three_step_builder() -> TraceBuilder:
    builder = TraceBuilder().schedule(0).schedule(100).schedule(150)
    builder.gpu("deep_gemm_fp8", 10, 30, stream=7).gpu("nccl_all_reduce", 30, 30, stream=8).memcpy(70, 5)
    builder.gpu("elementwise", 110, 30)
    builder.gpu("elementwise", 210, 10)
    return builder


def test_phase_averages_match_hand_computation(tmp_path):
    summary = summarize(trace_file(tmp_path, three_step_builder()), Options(prefill_steps=1))
    assert [p.phase for p in summary.phases] == ["prefill", "decode"]
    prefill, decode = summary.phases

    assert (prefill.steps, prefill.step_time_mean, prefill.step_time_max) == (1, D(100), D(100))
    assert prefill.exclusive_mean == {"gemm.deepgemm": D(30), "comm.nccl": D(20), "memcpy": D(5)}
    assert prefill.inclusive_mean["comm.nccl"] == D(30)
    assert (prefill.busy_mean, prefill.idle_mean, prefill.overlap_mean) == (D(55), D(45), D(10))
    assert prefill.share("comm.nccl") == D("0.2")
    assert prefill.other_ratio == D(0)

    assert (decode.steps, decode.step_time_mean, decode.step_time_max) == (2, D(60), D(70))
    assert decode.exclusive_mean == {"other": D(20)}
    assert decode.idle_mean == D(40)
    assert decode.other_ratio == D(20) / D(60)


def test_idle_and_exclusive_shares_add_up_to_one(tmp_path):
    summary = summarize(trace_file(tmp_path, three_step_builder()), Options(prefill_steps=1))
    for phase in summary.phases:
        assert sum(phase.exclusive_mean.values()) + phase.idle_mean == phase.step_time_mean


def test_skip_steps_leaves_them_out_of_the_averages(tmp_path):
    summary = summarize(trace_file(tmp_path, three_step_builder()), Options(prefill_steps=1, skip_steps=1))
    assert [p.phase for p in summary.phases] == ["decode"]
    assert summary.phases[0].steps == 2


def test_phases_are_detected_from_launch_forward_when_not_given(tmp_path):
    builder = (
        three_step_builder()
        .launch_forward(5, 2, reqs=2, tokens=4096)
        .launch_forward(105, 2, reqs=64, tokens=64)
    )
    summary = summarize(trace_file(tmp_path, builder), Options())
    assert [(p.phase, p.steps) for p in summary.phases] == [("prefill", 1), ("decode", 1), ("unknown", 1)]


def test_unmatched_and_spill_ratios(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.kernel("a", 10, 10, launch_ts=5)
    builder.kernel("b", 90, 30, launch_ts=85)
    builder.kernel("c", 130, 10)
    summary = summarize(trace_file(tmp_path, builder), Options(prefill_steps=0))
    decode = summary.phases[0]
    assert decode.unmatched_ratio == D(10) / D(50)
    assert decode.spill_ratio == D(20) / D(40)


def test_ratios_are_none_when_nothing_can_be_measured(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).gpu("k", 150, 5)
    decode = summarize(trace_file(tmp_path, builder), Options(prefill_steps=0)).phases[0]
    assert decode.spill_ratio is None
    assert decode.unmatched_ratio == D(1)


def test_bubble_average_is_reported(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).bubble(10, 4).bubble(110, 8).gpu("k", 199, 1)
    assert summarize(trace_file(tmp_path, builder), Options(prefill_steps=0)).phases[0].bubble_mean == D(6)


def test_a_device_the_trace_does_not_have_is_an_analysis_error(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).gpu("k", 10, 5)
    with pytest.raises(AnalysisError, match="device 5.*\\[0\\]"):
        summarize(trace_file(tmp_path, builder), Options(prefill_steps=0, device=5))


def test_trace_without_any_gpu_events_warns_and_is_all_idle(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    summary = summarize(trace_file(tmp_path, builder), Options(prefill_steps=0, device=3))
    assert summary.warnings == [
        "the trace has no GPU events, so every step shows 100% idle (a non-CUDA trace?)"
    ]
    assert summary.phases[0].idle_mean == summary.phases[0].step_time_mean


def test_unsplit_phases_are_flagged_with_the_remedy(tmp_path):
    summary = summarize(trace_file(tmp_path, three_step_builder()), Options())
    assert any("--prefill-steps" in w for w in summary.warnings)
    assert not any(
        "--prefill-steps" in w
        for w in summarize(trace_file(tmp_path, three_step_builder()), Options(prefill_steps=1)).warnings
    )


def test_loader_warnings_are_carried_through(tmp_path):
    builder = three_step_builder().raw(
        {"ph": "X", "cat": "kernel", "name": "k", "pid": 0, "ts": "NaN", "dur": 1}
    )
    summary = summarize(trace_file(tmp_path, builder), Options(prefill_steps=1))
    assert any("skipped 1" in w for w in summary.warnings)


def test_unclassified_kernels_are_ranked_by_total_duration(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.gpu("mystery_a", 1, 10).gpu("mystery_b", 20, 30).gpu("mystery_a", 60, 25).gpu("deep_gemm", 90, 5)
    builder.gpu("tail", 99, 1)
    summary = summarize(trace_file(tmp_path, builder), Options(prefill_steps=0, show_unclassified=2))
    assert summary.unclassified == [("mystery_a", D(35)), ("mystery_b", D(30))]


def test_unclassified_is_empty_unless_requested(tmp_path):
    summary = summarize(trace_file(tmp_path, three_step_builder()), Options(prefill_steps=1))
    assert summary.unclassified is None


def test_user_rules_change_the_categories(tmp_path):
    rules = tmp_path / "rules.toml"
    rules.write_text('[[rule]]\ncategory = "mine"\npattern = "elementwise"\n')
    summary = summarize(
        trace_file(tmp_path, three_step_builder()),
        Options(prefill_steps=1, rules=str(rules), no_default_rules=True),
    )
    assert summary.phases[1].exclusive_mean == {"mine": D(20)}
    assert summary.phases[0].exclusive_mean == {"other": D(50), "memcpy": D(5)}


def unclassified(tmp_path, builder, count=5, **options):
    summary = summarize(
        trace_file(tmp_path, builder), Options(prefill_steps=0, show_unclassified=count, **options)
    )
    return summary.unclassified


def test_unclassified_ties_are_broken_by_name(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200).gpu("b", 110, 7).gpu("a", 130, 7).gpu("z", 150, 9)
    assert [name for name, _ in unclassified(tmp_path, builder)] == ["z", "a", "b"]


def test_unclassified_only_counts_the_chosen_device_and_the_analysed_span(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200)
    builder.gpu("other_device", 110, 5, device=1).gpu("before_first_step", 50, 5)
    builder.gpu("on_the_first_boundary", 100, 3).gpu("kept", 150, 4)
    assert unclassified(tmp_path, builder) == [("kept", D(4)), ("on_the_first_boundary", D(3))]
    assert unclassified(tmp_path, builder, device=1) == [("other_device", D(5))]


def test_unclassified_is_empty_when_everything_is_classified(tmp_path):
    builder = TraceBuilder().schedule(0).schedule(100).gpu("deep_gemm", 10, 5).gpu("nccl", 20, 5)
    assert unclassified(tmp_path, builder) == []
