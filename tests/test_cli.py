import subprocess
import sys

import pytest

from deepinfer_trace_analyzer import cli
from deepinfer_trace_analyzer.errors import AnalysisError, InputError


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "deepinfer_trace_analyzer", *args], capture_output=True, text=True
    )


def test_help_lists_both_subcommands():
    result = run("--help")
    assert result.returncode == 0
    assert "summary" in result.stdout and "diff" in result.stdout


@pytest.mark.parametrize(("error", "code"), [(InputError("bad file"), 1), (AnalysisError("bad trace"), 2)])
def test_errors_map_to_exit_codes_and_stderr(monkeypatch, capsys, error, code):
    def boom(args):
        raise error

    monkeypatch.setattr(cli, "_run_summary", boom)
    assert cli.main(["summary", "x.json"]) == code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(error) in captured.err
