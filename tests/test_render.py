import json
import os
from decimal import Decimal as D
from pathlib import Path

import pytest

from builders import TraceBuilder
from deepinfer_trace_analyzer import cli
from deepinfer_trace_analyzer.render import render_summary
from deepinfer_trace_analyzer.rules import Classifier
from deepinfer_trace_analyzer.summary import Options, summarize

GOLDEN = Path(__file__).parent / "golden"


def trace_path(tmp_path) -> Path:
    builder = TraceBuilder().schedule(0).schedule(100).schedule(150)
    builder.gpu("deep_gemm_fp8", 10, 30, stream=7).gpu("nccl_all_reduce", 30, 30, stream=8).memcpy(70, 5)
    builder.gpu("elementwise", 110, 30).gpu("mystery_kernel", 210, 10)
    return builder.write(tmp_path / "t.json")


def summary_for(tmp_path, **options):
    return summarize(trace_path(tmp_path), Options(prefill_steps=1, **options))


def check_golden(name: str, actual: str) -> None:
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(actual, encoding="utf-8")
    assert actual == path.read_text(encoding="utf-8")


def test_table_matches_golden(tmp_path):
    check_golden(
        "summary_table.txt", render_summary(summary_for(tmp_path), "table").replace(str(tmp_path), "TMP")
    )


def test_markdown_matches_golden(tmp_path):
    check_golden(
        "summary_markdown.md", render_summary(summary_for(tmp_path), "markdown").replace(str(tmp_path), "TMP")
    )


def test_table_with_unclassified_kernels_matches_golden(tmp_path):
    text = render_summary(summary_for(tmp_path, show_unclassified=5), "table").replace(str(tmp_path), "TMP")
    check_golden("summary_table_unclassified.txt", text)


def test_json_carries_schema_options_rules_and_per_step_values(tmp_path):
    document = json.loads(render_summary(summary_for(tmp_path), "json"))
    assert document["schema_version"] == 1
    assert document["trace"].endswith("t.json")
    assert document["device"] == 0
    assert document["options"]["prefill_steps"] == 1
    assert document["rules"][0] == {"category": "comm.nccl", "pattern": "nccl"}
    first = document["steps"][0]
    assert (first["index"], first["phase"], D(first["start"]), D(first["end"])) == (0, "prefill", 0, 100)
    assert D(first["exclusive"]["gemm.deepgemm"]) == 30 and D(first["idle"]) == 45
    assert D(first["inclusive"]["comm.nccl"]) == 30


def test_json_phase_means_are_the_average_of_the_step_values(tmp_path):
    document = json.loads(render_summary(summary_for(tmp_path), "json"))
    decode_steps = [s for s in document["steps"] if s["phase"] == "decode"]
    decode = next(p for p in document["phases"] if p["phase"] == "decode")
    assert D(decode["exclusive_mean"]["other"]) == sum(
        D(s["exclusive"].get("other", "0")) for s in decode_steps
    ) / len(decode_steps)
    assert D(decode["step_time_mean"]) == sum(D(s["step_time"]) for s in decode_steps) / len(decode_steps)


def test_json_keeps_microsecond_precision(tmp_path):
    builder = (
        TraceBuilder().schedule("1000.000001").schedule("1100.000002").gpu("k", "1010.000003", "20.000004")
    )
    path = builder.write(tmp_path / "p.json", stringify=True)
    document = json.loads(render_summary(summarize(path, Options(prefill_steps=0)), "json"))
    assert document["steps"][0]["inclusive"]["other"] == "20.000004"
    assert document["steps"][0]["start"] == "1000.000001"


def test_classifier_exposes_rules_user_first(tmp_path):
    rules = tmp_path / "r.toml"
    rules.write_text('[[rule]]\ncategory = "mine"\npattern = "x"\n')
    assert Classifier.from_file(rules).rules()[0] == ("mine", "x")


def run_cli(capsys, *args):
    code = cli.main(list(args))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_summary_json_goes_to_stdout_with_a_clean_stderr(tmp_path, capsys):
    code, out, err = run_cli(
        capsys, "summary", str(trace_path(tmp_path)), "--prefill-steps", "1", "--format", "json"
    )
    assert code == 0 and err == ""
    assert json.loads(out)["schema_version"] == 1


def test_cli_warnings_go_to_stderr_only(tmp_path, capsys):
    path = TraceBuilder().schedule(0).schedule(100).write(tmp_path / "e.json")
    code, out, err = run_cli(capsys, "summary", str(path), "--prefill-steps", "0", "--format", "json")
    assert code == 0
    assert "warning:" in err and "no GPU events" in err
    assert "warning:" not in out
    json.loads(out)


def test_cli_markdown_and_unclassified(tmp_path, capsys):
    path = str(trace_path(tmp_path))
    code, out, _ = run_cli(
        capsys, "summary", path, "--prefill-steps", "1", "--format", "markdown", "--show-unclassified"
    )
    assert code == 0 and "| category |" in out and "mystery_kernel" in out


def test_cli_rules_flags_reach_the_classifier(tmp_path, capsys):
    rules = tmp_path / "r.toml"
    rules.write_text('[[rule]]\ncategory = "mine"\npattern = "elementwise"\n')
    path = str(trace_path(tmp_path))
    code, out, _ = run_cli(
        capsys,
        "summary",
        path,
        "--prefill-steps",
        "1",
        "--rules",
        str(rules),
        "--no-default-rules",
        "--format",
        "json",
    )
    assert code == 0
    decode = next(p for p in json.loads(out)["phases"] if p["phase"] == "decode")
    assert "mine" in decode["exclusive_mean"]


@pytest.mark.parametrize(
    ("make_args", "code"),
    [
        (lambda tmp: ["summary", str(tmp / "missing.json")], 1),
        (lambda tmp: ["summary", str(trace_path(tmp)), "--rules", str(tmp / "missing.toml")], 1),
        (lambda tmp: ["summary", str(trace_path(tmp)), "--prefill-steps", "99"], 2),
    ],
)
def test_cli_errors_exit_nonzero_with_empty_stdout(tmp_path, capsys, make_args, code):
    actual, out, err = run_cli(capsys, *make_args(tmp_path))
    assert (actual, out) == (code, "")
    assert err.startswith("error:")


def test_cli_invalid_rules_file_exits_two(tmp_path, capsys):
    rules = tmp_path / "bad.toml"
    rules.write_text("not = [valid")
    actual, out, _ = run_cli(capsys, "summary", str(trace_path(tmp_path)), "--rules", str(rules))
    assert (actual, out) == (2, "")


def test_json_phase_and_step_documents_match_hand_computation(tmp_path):
    document = json.loads(render_summary(summary_for(tmp_path), "json"))
    assert document["unit"] == "us" and document["device"] == 0 and document["dropped_gpu_us"] == "0"
    prefill, decode = document["phases"]
    assert prefill == {
        "phase": "prefill", "steps": 1, "step_time_mean": "100", "step_time_max": "100", "busy_mean": "55",
        "idle_mean": "45", "overlap_mean": "10", "bubble_mean": "0",
        "exclusive_mean": {"comm.nccl": "20", "gemm.deepgemm": "30", "memcpy": "5"},
        "inclusive_mean": {"comm.nccl": "30", "gemm.deepgemm": "30", "memcpy": "5"},
        "other_ratio": "0", "unmatched_ratio": "1", "spill_ratio": None,
    }  # fmt: skip
    assert (decode["steps"], decode["step_time_mean"], decode["step_time_max"]) == (2, "60", "70")
    assert (decode["busy_mean"], decode["idle_mean"], decode["overlap_mean"]) == ("20", "40", "0")
    assert decode["exclusive_mean"] == {"other": "20"} and D(decode["other_ratio"]) == D(20) / D(60)

    first, second, third = document["steps"]
    assert first == {
        "index": 0, "phase": "prefill", "start": "0", "end": "100", "step_time": "100", "busy_union": "55",
        "idle": "45", "overlap": "10", "bubble": "0", "gpu_us": "65", "matched_us": "0", "spill_us": "0",
        "unmatched_us": "65",
        "exclusive": {"comm.nccl": "20", "gemm.deepgemm": "30", "memcpy": "5"},
        "inclusive": {"comm.nccl": "30", "gemm.deepgemm": "30", "memcpy": "5"},
    }  # fmt: skip
    assert (second["start"], second["end"], second["busy_union"], second["idle"]) == (
        "100",
        "150",
        "30",
        "20",
    )
    assert (third["start"], third["end"], third["busy_union"], third["idle"]) == ("150", "220", "10", "60")
    assert third["exclusive"] == {"other": "10"} and third["unmatched_us"] == "10"


def test_json_options_and_dropped_time_are_reported(tmp_path):
    builder = TraceBuilder().schedule(100).schedule(200).gpu("early", 80, 40).gpu("k", 150, 5)
    path = builder.write(tmp_path / "d.json")
    document = json.loads(render_summary(summarize(path, Options(prefill_steps=0, device=0)), "json"))
    assert document["dropped_gpu_us"] == "20"
    assert document["options"] == {
        "prefill_steps": 0, "prefill_threshold": "16", "skip_steps": 0, "rules": None,
        "no_default_rules": False, "device": 0, "show_unclassified": None,
    }  # fmt: skip


@pytest.mark.parametrize("fmt", ["table", "markdown"])
def test_text_formats_report_gpu_time_before_the_first_step(tmp_path, fmt):
    builder = TraceBuilder().schedule(100).schedule(200).gpu("early", 80, 40).gpu("k", 150, 5)
    path = builder.write(tmp_path / "d.json")
    assert "GPU time before the first step: 20.000 us" in render_summary(
        summarize(path, Options(prefill_steps=0)), fmt
    )


def test_markdown_tables_have_one_separator_cell_per_column(tmp_path):
    lines = render_summary(summary_for(tmp_path), "markdown").splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| category |"))
    assert lines[header].count("|") == lines[header + 1].count("|") == lines[header + 2].count("|")
    assert lines[header + 1] == "|---|--:|--:|--:|"


@pytest.mark.parametrize(("fmt", "expected"), [("table", "  none"), ("markdown", "- none")])
def test_an_empty_unclassified_list_says_none(tmp_path, fmt, expected):
    builder = TraceBuilder().schedule(0).schedule(100).gpu("deep_gemm", 10, 5).gpu("nccl", 20, 5)
    path = builder.write(tmp_path / "c.json")
    text = render_summary(summarize(path, Options(prefill_steps=0, show_unclassified=3)), fmt)
    assert expected in text.splitlines()


def test_unclassified_json_lists_names_and_durations(tmp_path):
    document = json.loads(render_summary(summary_for(tmp_path, show_unclassified=1), "json"))
    assert document["unclassified"] == [{"name": "elementwise", "duration_us": "30"}]


def test_json_numbers_never_use_exponent_notation(tmp_path):
    text = render_summary(summary_for(tmp_path), "json")
    assert "E-" not in text and "E+" not in text


def cli_json(capsys, tmp_path, *args):
    code, out, _ = run_cli(capsys, "summary", str(trace_path(tmp_path)), "--format", "json", *args)
    assert code == 0
    return json.loads(out)


def test_cli_skip_steps_leaves_leading_steps_out(tmp_path, capsys):
    document = cli_json(capsys, tmp_path, "--prefill-steps", "1", "--skip-steps", "1")
    assert [s["index"] for s in document["steps"]] == [1, 2]


def test_cli_prefill_threshold_moves_the_phase_boundary(tmp_path, capsys):
    builder = TraceBuilder().schedule(0).schedule(100).launch_forward(5, 2, reqs=2, tokens=40).gpu("k", 50, 5)
    path = str(builder.write(tmp_path / "t.json"))
    phases = lambda *extra: [  # noqa: E731
        s["phase"]
        for s in json.loads(run_cli(capsys, "summary", path, "--format", "json", *extra)[1])["steps"]
    ]
    assert phases() == ["prefill", "unknown"]
    assert phases("--prefill-threshold", "25") == ["decode", "unknown"]


def test_cli_no_default_rules_without_a_file_leaves_everything_other(tmp_path, capsys):
    document = cli_json(capsys, tmp_path, "--prefill-steps", "1", "--no-default-rules")
    assert document["rules"] == []
    assert set(document["phases"][0]["exclusive_mean"]) == {"other", "memcpy"}


def test_cli_device_selects_that_devices_events(tmp_path, capsys):
    builder = TraceBuilder().schedule(0).schedule(100).gpu("deep_gemm", 10, 20).gpu("nccl", 10, 50, device=1)
    path = str(builder.write(tmp_path / "t.json"))
    code, out, _ = run_cli(
        capsys, "summary", path, "--prefill-steps", "0", "--device", "1", "--format", "json"
    )
    assert code == 0 and json.loads(out)["phases"][0]["exclusive_mean"] == {"comm.nccl": "25"}


@pytest.mark.parametrize(
    "extra",
    [
        ["--prefill-threshold", "nan"],
        ["--prefill-threshold", "inf"],
        ["--prefill-threshold", "-1"],
        ["--prefill-threshold", "0"],
        ["--show-unclassified", "0"],
        ["--show-unclassified", "-2"],
        ["--device", "5"],
        ["--device", "-1"],
    ],
)
def test_cli_rejects_values_that_would_give_a_misleading_result(tmp_path, capsys, extra):
    code, out, err = run_cli(capsys, "summary", str(trace_path(tmp_path)), "--prefill-steps", "1", *extra)
    assert (code, out) == (2, "") and err.startswith("error:")
