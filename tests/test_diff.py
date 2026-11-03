import json
from decimal import Decimal as D
from pathlib import Path

import pytest
from builders import TraceBuilder

from deepinfer_trace_analyzer import cli
from deepinfer_trace_analyzer.diff import compare
from deepinfer_trace_analyzer.render import render_diff
from deepinfer_trace_analyzer.summary import Options, summarize

GOLDEN = Path(__file__).parent / "golden"


def launched_memcpy(builder: TraceBuilder, ts, dur, launch_ts) -> TraceBuilder:
    corr = builder.corr()
    builder.launch(launch_ts, corr=corr, name="cudaMemcpyAsync")
    return builder.gpu("Memcpy HtoD", ts, dur, cat="gpu_memcpy", corr=corr)


def before_builder() -> TraceBuilder:
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.kernel("deep_gemm", 10, 30, launch_ts=5).kernel("nccl_all_reduce", 30, 30, launch_ts=6, stream=8)
    launched_memcpy(builder, 70, 5, 65)
    builder.kernel("deep_gemm", 110, 20, launch_ts=105).kernel("deep_gemm", 140, 10, launch_ts=135)
    return builder


def after_builder() -> TraceBuilder:
    builder = TraceBuilder().schedule(0).schedule(100)
    builder.kernel("deep_gemm", 10, 30, launch_ts=5).kernel("nccl_all_reduce", 20, 20, launch_ts=6, stream=8)
    launched_memcpy(builder, 70, 5, 65)
    builder.kernel("deep_gemm", 110, 15, launch_ts=105).kernel("deep_gemm", 130, 10, launch_ts=125)
    return builder


def summary_of(tmp_path, builder, name, **options):
    path = builder.write(tmp_path / f"{name}.json")
    return summarize(path, Options(prefill_steps=1, **options))


def diff_of(tmp_path):
    return compare(summary_of(tmp_path, before_builder(), "before"), summary_of(tmp_path, after_builder(), "after"))


def rows(phase):
    return {row.name: row for row in phase.rows}


def test_exposed_comm_moves_to_idle_when_it_hides_under_compute(tmp_path):
    prefill = diff_of(tmp_path).phases[0]
    assert prefill.phase == "prefill"
    assert (prefill.step_time_before, prefill.step_time_after, prefill.step_time_delta) == (D(100), D(100), D(0))
    by_name = rows(prefill)
    assert (by_name["comm.nccl"].before, by_name["comm.nccl"].after, by_name["comm.nccl"].delta) == (D(20), D(0), D(-20))
    assert by_name["idle"].delta == D(20)
    assert by_name["gemm.deepgemm"].delta == D(0) and by_name["memcpy"].delta == D(0)


def test_share_changes_are_in_percentage_points(tmp_path):
    by_name = rows(diff_of(tmp_path).phases[0])
    assert (by_name["comm.nccl"].share_before, by_name["comm.nccl"].share_after) == (D("0.2"), D(0))
    assert by_name["comm.nccl"].share_delta_pt == D(-20)
    assert by_name["idle"].share_delta_pt == D(20)


def test_rows_are_ordered_by_absolute_change_then_name(tmp_path):
    names = [row.name for row in diff_of(tmp_path).phases[0].rows]
    assert names == ["comm.nccl", "idle", "gemm.deepgemm", "memcpy"]


def test_inclusive_comm_barely_changes_while_overlap_grows(tmp_path):
    prefill = diff_of(tmp_path).phases[0]
    nccl = next(row for row in prefill.inclusive_rows if row.name == "comm.nccl")
    assert (nccl.before, nccl.after, nccl.delta) == (D(30), D(20), D(-10))
    assert (prefill.overlap.before, prefill.overlap.after, prefill.overlap.delta) == (D(10), D(20), D(10))


def test_step_time_change_is_fully_explained_by_the_rows(tmp_path):
    for phase in diff_of(tmp_path).phases:
        assert sum(row.delta for row in phase.rows) == phase.step_time_delta
        assert phase.residual == D(0)
    decode = diff_of(tmp_path).phases[1]
    assert (decode.step_time_before, decode.step_time_after, decode.step_time_delta) == (D(50), D(40), D(-10))
    assert rows(decode)["gemm.deepgemm"].delta == D(-5) and rows(decode)["idle"].delta == D(-5)


def test_diffing_a_trace_with_itself_changes_nothing_and_warns_about_nothing(tmp_path):
    summary = summary_of(tmp_path, before_builder(), "same")
    result = compare(summary, summary)
    assert result.warnings == []
    for phase in result.phases:
        assert phase.step_time_delta == 0 and phase.residual == 0 and phase.overlap.delta == 0
        assert all(row.delta == 0 and row.share_delta_pt == 0 for row in phase.rows)
        assert all(row.delta == 0 for row in phase.inclusive_rows)


def test_per_step_means_make_traces_of_different_length_comparable(tmp_path):
    longer = before_builder().schedule(200).kernel("deep_gemm", 210, 20, launch_ts=205).kernel("tail", 240, 10, launch_ts=235)
    a = summary_of(tmp_path, before_builder(), "a")
    b = summary_of(tmp_path, longer, "b")
    decode = compare(a, b).phases[1]
    assert (decode.steps_before, decode.steps_after) == (1, 2)
    assert any("decode" in w and "1" in w and "2" in w for w in compare(a, b).warnings)
    assert decode.step_time_before == D(50) and decode.step_time_after == D(75)


def test_warns_when_a_trace_is_poorly_matched_or_poorly_classified(tmp_path):
    messy = TraceBuilder().schedule(0).schedule(100).gpu("mystery", 10, 50).gpu("mystery2", 110, 30)
    clean = summary_of(tmp_path, before_builder(), "clean")
    warnings = compare(clean, summary_of(tmp_path, messy, "messy")).warnings
    assert any("unmatched" in w for w in warnings)
    assert any("other" in w for w in warnings)


def test_warns_when_the_device_count_differs(tmp_path):
    two_devices = before_builder().gpu("nccl", 20, 5, device=1)
    warnings = compare(summary_of(tmp_path, before_builder(), "x"), summary_of(tmp_path, two_devices, "y")).warnings
    assert any("device" in w for w in warnings)


def test_a_phase_present_in_only_one_trace_is_reported_not_compared(tmp_path):
    decode_only = TraceBuilder().schedule(0).schedule(100).kernel("deep_gemm", 10, 30, launch_ts=5)
    before = summary_of(tmp_path, before_builder(), "x")
    after = summarize(decode_only.write(tmp_path / "y.json"), Options(prefill_steps=0))
    result = compare(before, after)
    assert [p.phase for p in result.phases] == ["decode"]
    assert any("prefill" in w and "only" in w for w in result.warnings)


def test_table_matches_golden(tmp_path):
    import os

    text = render_diff(diff_of(tmp_path), "table").replace(str(tmp_path), "TMP")
    path = GOLDEN / "diff_table.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")


def test_json_has_both_traces_phases_and_exact_deltas(tmp_path):
    document = json.loads(render_diff(diff_of(tmp_path), "json"))
    assert document["schema_version"] == 1
    assert document["before"].endswith("before.json") and document["after"].endswith("after.json")
    prefill = document["phases"][0]
    nccl = next(r for r in prefill["rows"] if r["name"] == "comm.nccl")
    assert (D(nccl["before"]), D(nccl["after"]), D(nccl["delta"]), D(nccl["share_delta_pt"])) == (20, 0, -20, -20)
    assert D(prefill["residual"]) == 0


def test_markdown_is_a_table(tmp_path):
    text = render_diff(diff_of(tmp_path), "markdown")
    assert "| category |" in text and "comm.nccl" in text


def run_cli(capsys, *args):
    code = cli.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_diff_json(tmp_path, capsys):
    before = before_builder().write(tmp_path / "b.json")
    after = after_builder().write(tmp_path / "a.json")
    code, out, err = run_cli(capsys, "diff", str(before), str(after), "--prefill-steps", "1", "--format", "json")
    assert code == 0 and err == ""
    assert json.loads(out)["phases"][0]["phase"] == "prefill"


def test_cli_diff_prints_warnings_to_stderr(tmp_path, capsys):
    before = before_builder().write(tmp_path / "b.json")
    messy = TraceBuilder().schedule(0).schedule(100).gpu("mystery", 10, 50).write(tmp_path / "m.json")
    code, out, err = run_cli(capsys, "diff", str(before), str(messy), "--prefill-steps", "1")
    assert code == 0 and "warning:" in err and "warning:" not in out


@pytest.mark.parametrize("which", ["before", "after"])
def test_cli_diff_missing_input_exits_one_with_empty_stdout(tmp_path, capsys, which):
    good = before_builder().write(tmp_path / "g.json")
    missing = str(tmp_path / "missing.json")
    args = [missing, str(good)] if which == "before" else [str(good), missing]
    code, out, err = run_cli(capsys, "diff", *args)
    assert (code, out) == (1, "") and err.startswith("error:")
