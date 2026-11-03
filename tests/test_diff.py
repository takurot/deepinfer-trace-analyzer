import json
from decimal import Decimal as D
from pathlib import Path

import pytest

from builders import TraceBuilder
from deepinfer_trace_analyzer import cli
from deepinfer_trace_analyzer.diff import compare
from deepinfer_trace_analyzer.render_diff import render_diff
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
    return compare(
        summary_of(tmp_path, before_builder(), "before"), summary_of(tmp_path, after_builder(), "after")
    )


def rows(phase):
    return {row.name: row for row in phase.rows}


def test_exposed_comm_moves_to_idle_when_it_hides_under_compute(tmp_path):
    prefill = diff_of(tmp_path).phases[0]
    assert prefill.phase == "prefill"
    assert (prefill.step_time_before, prefill.step_time_after, prefill.step_time_delta) == (
        D(100),
        D(100),
        D(0),
    )
    by_name = rows(prefill)
    assert (by_name["comm.nccl"].before, by_name["comm.nccl"].after, by_name["comm.nccl"].delta) == (
        D(20),
        D(0),
        D(-20),
    )
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
    longer = (
        before_builder()
        .schedule(200)
        .kernel("deep_gemm", 210, 20, launch_ts=205)
        .kernel("tail", 240, 10, launch_ts=235)
    )
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
    warnings = compare(
        summary_of(tmp_path, before_builder(), "x"), summary_of(tmp_path, two_devices, "y")
    ).warnings
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
    nccl = next(r for r in prefill["rows"] if r["category"] == "comm.nccl")
    assert (D(nccl["before"]), D(nccl["after"]), D(nccl["delta"]), D(nccl["share_delta_pt"])) == (
        20,
        0,
        -20,
        -20,
    )
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
    code, out, err = run_cli(
        capsys, "diff", str(before), str(after), "--prefill-steps", "1", "--format", "json"
    )
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


def single_step(tmp_path, name, *, matched, unmatched="0", other="0"):
    """One 100 us step: matched GEMM, GEMM with no launch, and a matched kernel that no rule knows."""
    builder = TraceBuilder().schedule(100)
    start = D(100)
    for dur, kind in ((matched, "matched"), (unmatched, "unmatched"), (other, "other")):
        dur = D(dur)
        if dur == 0:
            continue
        if kind == "unmatched":
            builder.gpu("deep_gemm", start, dur)
        else:
            builder.kernel("deep_gemm" if kind == "matched" else "mystery", start, dur, launch_ts=99)
        start += dur
    path = builder.write(tmp_path / f"{name}.json")
    return summarize(path, Options(prefill_steps=0))


def test_other_share_warns_only_above_ten_percent(tmp_path):
    clean = single_step(tmp_path, "clean", matched="100")
    exactly = single_step(tmp_path, "ten", matched="90", other="10")
    above = single_step(tmp_path, "above", matched="89.9", other="10.1")
    assert compare(clean, exactly).warnings == []
    assert compare(clean, above).warnings == [
        "after decode: 10% of step time is 'other'; try --show-unclassified or --rules"
    ]


def test_unmatched_share_warns_only_above_five_percent(tmp_path):
    clean = single_step(tmp_path, "clean", matched="100")
    exactly = single_step(tmp_path, "five", matched="95", unmatched="5")
    above = single_step(tmp_path, "above", matched="94.9", unmatched="5.1")
    assert compare(clean, exactly).warnings == []
    assert compare(clean, above).warnings == ["after decode: 5% of GPU time is unmatched (no launch)"]


def test_warnings_name_the_side_they_come_from(tmp_path):
    messy = TraceBuilder().schedule(0).gpu("mystery", 0, 30).gpu("deep_gemm", 30, 70)
    messy = summarize(messy.write(tmp_path / "messy.json"), Options(prefill_steps=0))
    clean = single_step(tmp_path, "clean", matched="100")
    expected = {
        "decode: 100% of GPU time is unmatched (no launch)",
        "decode: 30% of step time is 'other'; try --show-unclassified or --rules",
    }
    assert set(compare(messy, clean).warnings) == {f"before {w}" for w in expected}
    assert set(compare(clean, messy).warnings) == {f"after {w}" for w in expected}


def test_a_phase_missing_on_one_side_names_that_side(tmp_path):
    both = summary_of(tmp_path, before_builder(), "both")
    decode_only = summarize(
        TraceBuilder()
        .schedule(0)
        .schedule(100)
        .kernel("deep_gemm", 10, 30, launch_ts=5)
        .write(tmp_path / "d.json"),
        Options(prefill_steps=0),
    )
    assert "phase prefill exists only in BEFORE, so it is not compared" in compare(both, decode_only).warnings
    assert "phase prefill exists only in AFTER, so it is not compared" in compare(decode_only, both).warnings


def test_input_warnings_are_passed_through_with_their_side(tmp_path):
    bad = before_builder().raw({"ph": "X", "cat": "kernel", "name": "k", "pid": 0, "ts": "NaN", "dur": 1})
    result = compare(summary_of(tmp_path, bad, "bad"), summary_of(tmp_path, before_builder(), "ok"))
    assert [w for w in result.warnings if w.startswith("before: skipped 1")]
    assert not [w for w in result.warnings if w.startswith("after:")]


def test_categories_that_exist_on_one_side_only_get_a_zero_on_the_other(tmp_path):
    plain = TraceBuilder().schedule(0).schedule(100)
    plain.kernel("deep_gemm", 10, 30, launch_ts=5).kernel("tail", 150, 5, launch_ts=140)
    with_copy = TraceBuilder().schedule(0).schedule(100)
    with_copy.kernel("deep_gemm", 10, 30, launch_ts=5).kernel("tail", 150, 5, launch_ts=140)
    launched_memcpy(with_copy, 60, 5, 55)
    result = compare(summary_of(tmp_path, plain, "plain"), summary_of(tmp_path, with_copy, "copy"))
    memcpy = next(r for r in result.phases[0].rows if r.name == "memcpy")
    assert (memcpy.before, memcpy.after, memcpy.delta) == (D(0), D(5), D(5))
    inclusive = next(r for r in result.phases[0].inclusive_rows if r.name == "memcpy")
    assert (inclusive.before, inclusive.after) == (D(0), D(5))
    reverse = compare(summary_of(tmp_path, with_copy, "copy"), summary_of(tmp_path, plain, "plain"))
    assert next(r for r in reverse.phases[0].rows if r.name == "memcpy").delta == D(-5)


def test_equal_changes_are_ordered_by_name(tmp_path):
    visible = TraceBuilder().schedule(0).schedule(100).kernel("deep_gemm", 10, 30, launch_ts=5)
    launched_memcpy(visible, 70, 5, 65)
    hidden = TraceBuilder().schedule(0).schedule(100).kernel("deep_gemm", 10, 30, launch_ts=5)
    launched_memcpy(hidden, 20, 5, 15)
    hidden.kernel("tail", 150, 1, launch_ts=140)
    visible.kernel("tail", 150, 1, launch_ts=140)
    phase = compare(summary_of(tmp_path, visible, "v"), summary_of(tmp_path, hidden, "h")).phases[0]
    assert [(r.name, r.delta) for r in phase.rows[:2]] == [("idle", D(5)), ("memcpy", D(-5))]


def thirds(tmp_path, name, extra):
    builder = TraceBuilder().schedule(0).schedule(10).schedule(20)
    builder.kernel("deep_gemm", 1, 3 + extra, launch_ts=0).kernel("deep_gemm", 11, 3, launch_ts=10)
    builder.kernel("deep_gemm", 21, 3, launch_ts=20).kernel("tail", 30, 1, launch_ts=29)
    return summary_of(tmp_path, builder, name)


def test_the_residual_is_zero_and_prints_cleanly_when_means_do_not_terminate(tmp_path):
    result = compare(thirds(tmp_path, "a", 0), thirds(tmp_path, "b", 2))
    assert all(phase.residual == 0 for phase in result.phases)
    text = render_diff(result, "table")
    assert "-0.000" not in text and "residual 0.000 us" in text
    document = json.loads(render_diff(result, "json"))
    assert all(p["residual"] == "0" for p in document["phases"])
    assert "E-" not in render_diff(result, "json")


def test_headline_names_the_largest_changes(tmp_path):
    for fmt in ("table", "markdown"):
        text = render_diff(diff_of(tmp_path), fmt)
        assert "largest change: comm.nccl -20.000 us, idle (no GPU work) +20.000 us" in text
        assert "largest change: gemm.deepgemm -5.000 us, idle (no GPU work) -5.000 us" in text


def test_headline_says_none_when_nothing_changed(tmp_path):
    summary = summary_of(tmp_path, before_builder(), "same")
    assert "largest change: none" in render_diff(compare(summary, summary), "table")


def test_inclusive_tables_have_only_the_columns_that_apply(tmp_path):
    text = render_diff(diff_of(tmp_path), "table")
    inclusive_header = next(
        line for line in text.splitlines() if line.strip().startswith("category") and "% of step" not in line
    )
    assert inclusive_header.split() == ["category", "before", "us", "after", "us", "delta", "us"]


def test_the_legend_defines_exclusive_inclusive_and_overlap(tmp_path):
    for fmt in ("table", "markdown"):
        text = render_diff(diff_of(tmp_path), fmt)
        assert "exclusive" in text and "inclusive" in text and "double-count" in text


def test_markdown_tables_are_well_formed_and_match_golden(tmp_path):
    import os

    text = render_diff(diff_of(tmp_path), "markdown").replace(str(tmp_path), "TMP")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            assert lines[i - 1].count("|") == line.count("|") == lines[i + 1].count("|")
    assert "|---|--:|--:|--:|--:|--:|--:|" in lines and "|---|--:|--:|--:|" in lines
    path = GOLDEN / "diff_markdown.md"
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(text, encoding="utf-8")
    assert text == path.read_text(encoding="utf-8")


def test_json_document_matches_hand_computation(tmp_path):
    document = json.loads(render_diff(diff_of(tmp_path), "json"))
    assert (document["schema_version"], document["unit"], document["device"], document["warnings"]) == (
        1,
        "us",
        0,
        [],
    )
    assert document["options"]["prefill_steps"] == 1 and document["rules"][0] == {
        "category": "comm.nccl",
        "pattern": "nccl",
    }
    assert document["phases"][0] == {
        "phase": "prefill", "steps_before": 1, "steps_after": 1, "step_time_before": "100",
        "step_time_after": "100", "step_time_delta": "0",
        "rows": [
            {"category": "comm.nccl", "before": "20", "after": "0", "delta": "-20",
             "share_before": "0.2", "share_after": "0", "share_delta_pt": "-20"},
            {"category": "idle", "before": "45", "after": "65", "delta": "20",
             "share_before": "0.45", "share_after": "0.65", "share_delta_pt": "20"},
            {"category": "gemm.deepgemm", "before": "30", "after": "30", "delta": "0",
             "share_before": "0.3", "share_after": "0.3", "share_delta_pt": "0"},
            {"category": "memcpy", "before": "5", "after": "5", "delta": "0",
             "share_before": "0.05", "share_after": "0.05", "share_delta_pt": "0"},
        ],
        "inclusive_rows": [
            {"category": "comm.nccl", "before": "30", "after": "20", "delta": "-10"},
            {"category": "gemm.deepgemm", "before": "30", "after": "30", "delta": "0"},
            {"category": "memcpy", "before": "5", "after": "5", "delta": "0"},
        ],
        "overlap": {"category": "overlap", "before": "10", "after": "20", "delta": "10"},
        "residual": "0",
    }  # fmt: skip
    decode = document["phases"][1]
    assert (decode["step_time_before"], decode["step_time_after"], decode["step_time_delta"]) == (
        "50",
        "40",
        "-10",
    )


def test_cli_diff_keeps_before_and_after_in_order(tmp_path, capsys):
    before = before_builder().write(tmp_path / "b.json")
    after = after_builder().write(tmp_path / "a.json")
    code, out, _ = run_cli(
        capsys, "diff", str(before), str(after), "--prefill-steps", "1", "--format", "json"
    )
    document = json.loads(out)
    nccl = next(r for r in document["phases"][0]["rows"] if r["category"] == "comm.nccl")
    assert code == 0 and (nccl["before"], nccl["after"], nccl["delta"]) == ("20", "0", "-20")
    assert document["before"].endswith("b.json") and document["after"].endswith("a.json")


def cli_diff(capsys, tmp_path, *args):
    before = before_builder().write(tmp_path / "b.json")
    after = after_builder().write(tmp_path / "a.json")
    return run_cli(capsys, "diff", str(before), str(after), "--format", "json", *args)


def test_cli_diff_skip_steps_drops_the_leading_phase_from_both(tmp_path, capsys):
    code, out, _ = cli_diff(capsys, tmp_path, "--prefill-steps", "1", "--skip-steps", "1")
    assert code == 0 and [p["phase"] for p in json.loads(out)["phases"]] == ["decode"]


def test_cli_diff_no_default_rules_leaves_only_other_and_memcpy(tmp_path, capsys):
    code, out, _ = cli_diff(capsys, tmp_path, "--prefill-steps", "1", "--no-default-rules")
    document = json.loads(out)
    assert code == 0 and document["rules"] == []
    assert {r["category"] for r in document["phases"][0]["rows"]} == {"other", "memcpy", "idle"}


def test_cli_diff_prefill_threshold_changes_the_phase_split(tmp_path, capsys):
    builder = TraceBuilder().schedule(0).schedule(100).launch_forward(5, 2, reqs=2, tokens=40).gpu("k", 50, 5)
    path = str(builder.write(tmp_path / "t.json"))
    phases = lambda *extra: [  # noqa: E731
        p["phase"]
        for p in json.loads(run_cli(capsys, "diff", path, path, "--format", "json", *extra)[1])["phases"]
    ]
    assert phases() == ["prefill", "unknown"]
    assert phases("--prefill-threshold", "25") == ["decode", "unknown"]


def test_cli_diff_rejects_a_device_the_traces_do_not_have(tmp_path, capsys):
    code, out, err = cli_diff(capsys, tmp_path, "--prefill-steps", "1", "--device", "4")
    assert (code, out) == (2, "") and err.startswith("error:")
