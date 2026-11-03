from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from decimal import Decimal

from .diff import compare
from .errors import AnalyzerError
from .render import render_diff, render_summary
from .summary import Options, summarize


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


def _options(args: argparse.Namespace) -> Options:
    return Options(
        prefill_steps=args.prefill_steps,
        prefill_threshold=Decimal(str(args.prefill_threshold)),
        skip_steps=args.skip_steps,
        rules=args.rules,
        no_default_rules=args.no_default_rules,
        device=args.device,
        show_unclassified=getattr(args, "show_unclassified", None),
    )


def _run_summary(args: argparse.Namespace) -> str:
    summary = summarize(args.trace, _options(args))
    for warning in summary.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return render_summary(summary, args.format)


def _run_diff(args: argparse.Namespace) -> str:
    options = _options(args)
    result = compare(summarize(args.before, options), summarize(args.after, options))
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return render_diff(result, args.format)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = args.handler(args)
    except AnalyzerError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    sys.stdout.write(output)
    return 0
