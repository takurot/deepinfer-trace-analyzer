from decimal import Decimal

import pytest

from builders import TraceBuilder
from deepinfer_trace_analyzer.errors import AnalysisError
from deepinfer_trace_analyzer.loader import load_trace
from deepinfer_trace_analyzer.steps import split_steps


def loaded(tmp_path, builder):
    return load_trace(builder.write(tmp_path / "t.json"))


def windows(steps):
    return [(s.start, s.end) for s in steps]


def test_windows_run_from_one_schedule_to_the_next(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200).schedule(350)
    builder.gpu("k", 340, 60)
    steps = split_steps(loaded(tmp_path, builder), prefill_steps=0)
    assert windows(steps) == [
        (Decimal(100), Decimal(200)),
        (Decimal(200), Decimal(350)),
        (Decimal(350), Decimal(400)),
    ]
    assert [s.index for s in steps] == [0, 1, 2]


def test_last_window_ends_at_the_latest_event_end_across_cpu_and_gpu(tmp_path):
    cpu_late = TraceBuilder().schedule(100).schedule(200).bubble(210, 90).gpu("k", 220, 10)
    gpu_late = TraceBuilder().schedule(100).schedule(200).bubble(210, 10).gpu("k", 220, 90)
    assert split_steps(loaded(tmp_path, cpu_late), prefill_steps=0)[-1].end == Decimal(300)
    assert split_steps(loaded(tmp_path, gpu_late), prefill_steps=0)[-1].end == Decimal(310)


def test_last_window_is_never_negative_when_nothing_follows_the_schedule(tmp_path):
    builder = TraceBuilder().schedule(100, dur=0).schedule(200, dur=0)
    last = split_steps(loaded(tmp_path, builder), prefill_steps=0)[-1]
    assert last.end >= last.start


def test_prefill_steps_flag_marks_leading_steps(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200).schedule(300).schedule(400)
    steps = split_steps(loaded(tmp_path, builder), prefill_steps=2)
    assert [s.phase for s in steps] == ["prefill", "prefill", "decode", "decode"]


def test_prefill_steps_flag_wins_over_launch_forward_shape(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200)
    builder.launch_forward(110, 5, reqs=1, tokens=1).launch_forward(210, 5, reqs=1, tokens=4096)
    steps = split_steps(loaded(tmp_path, builder), prefill_steps=1)
    assert [s.phase for s in steps] == ["prefill", "decode"]


def test_auto_detects_phase_from_tokens_per_request(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200).schedule(300).schedule(400)
    builder.launch_forward(110, 5, reqs=2, tokens=8192)
    builder.launch_forward(210, 5, reqs=256, tokens=512)
    builder.launch_forward(310, 5, reqs=4, tokens=64)
    steps = split_steps(loaded(tmp_path, builder))
    assert [s.phase for s in steps] == ["prefill", "decode", "prefill", "unknown"]


def test_threshold_is_inclusive_and_configurable(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200)
    builder.launch_forward(110, 5, reqs=4, tokens=64).launch_forward(210, 5, reqs=4, tokens=60)
    default = split_steps(loaded(tmp_path, builder))
    assert [s.phase for s in default] == ["prefill", "decode"]
    strict = split_steps(loaded(tmp_path, builder), prefill_threshold=Decimal(17))
    assert [s.phase for s in strict] == ["decode", "decode"]


def test_zero_requests_is_unknown_not_a_crash(tmp_path):
    builder = TraceBuilder().schedule(100).launch_forward(110, 5, reqs=0, tokens=0)
    assert split_steps(loaded(tmp_path, builder))[0].phase == "unknown"


def test_launch_forward_outside_the_window_does_not_count(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200).schedule(300)
    builder.launch_forward(210, 5, reqs=1, tokens=4096)
    steps = split_steps(loaded(tmp_path, builder))
    assert [s.phase for s in steps] == ["unknown", "prefill", "unknown"]


def test_skip_steps_drops_leading_steps_but_keeps_phase_numbering(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200).schedule(300).schedule(400)
    steps = split_steps(loaded(tmp_path, builder), prefill_steps=2, skip_steps=1)
    assert [(s.index, s.phase) for s in steps] == [(1, "prefill"), (2, "decode"), (3, "decode")]


def test_prefill_decode_boundary_of_a_concatenated_trace_loses_no_time(tmp_path):
    # _rebase makes the decode segment start exactly where the prefill segment ends.
    builder = TraceBuilder().schedule(0).schedule(50).schedule(100).schedule(130)
    builder.gpu("prefill_last", 80, 20).gpu("decode_first", 100, 10)
    steps = split_steps(loaded(tmp_path, builder), prefill_steps=2)
    assert windows(steps)[1] == (Decimal(50), Decimal(100))
    assert windows(steps)[2][0] == Decimal(100)


def test_prefill_steps_beyond_step_count_is_an_analysis_error(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200)
    with pytest.raises(AnalysisError, match="--prefill-steps"):
        split_steps(loaded(tmp_path, builder), prefill_steps=3)


def test_skipping_every_step_is_an_analysis_error(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200)
    with pytest.raises(AnalysisError, match="--skip-steps"):
        split_steps(loaded(tmp_path, builder), skip_steps=2)
