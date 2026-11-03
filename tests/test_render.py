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
