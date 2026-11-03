from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .errors import AnalyzerError


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prefill-steps", type=int, metavar="N")
    parser.add_argument("--prefill-threshold", type=float, default=16.0, metavar="R")
    parser.add_argument("--skip-steps", type=int, default=0, metavar="K")
    parser.add_argument("--rules", metavar="FILE")
    parser.add_argument("--no-default-rules", action="store_true")
    parser.add_argument("--device", type=int, default=0, metavar="D")
    parser.add_argument("--format", choices=("table", "markdown", "json"), default="table")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepinfer-trace-analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary", help="summarize one trace by phase and kernel category")
    summary.add_argument("trace")
    _add_common(summary)
    summary.add_argument("--show-unclassified", nargs="?", type=int, const=20, default=None, metavar="K")
    summary.set_defaults(handler=_run_summary)

    diff = sub.add_parser("diff", help="compare two traces")
    diff.add_argument("before")
    diff.add_argument("after")
    _add_common(diff)
    diff.set_defaults(handler=_run_diff)
    return parser


def _run_summary(args: argparse.Namespace) -> str:
    raise NotImplementedError


def _run_diff(args: argparse.Namespace) -> str:
    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = args.handler(args)
    except AnalyzerError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    sys.stdout.write(output)
    return 0
