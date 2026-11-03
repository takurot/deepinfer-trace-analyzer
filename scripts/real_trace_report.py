"""Pass/fail report for a trace taken on real hardware (issue #11, SPEC section 6 criterion 2 and section 8).

    uv run python scripts/real_trace_report.py trace.json --prefill-steps 2

Exit status is 1 if a hard check fails (too much time in `other`), 0 otherwise. Warnings and
information lines are the things to read and write down in SPEC section 8.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from deepinfer_trace_analyzer.errors import AnalyzerError
from deepinfer_trace_analyzer.fmt import percent, us
from deepinfer_trace_analyzer.summary import Options, Summary, summarize

OTHER_LIMIT = Decimal("0.10")
UNMATCHED_LIMIT = Decimal("0.05")
LAST_STEP_BAND = (Decimal("0.8"), Decimal("1.2"))
GRAPH_LAUNCH = "cudaGraphLaunch"


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def evaluate(summary: Summary) -> list[Check]:
    checks = []
    for phase in summary.phases:
        other_ok = phase.other_ratio <= OTHER_LIMIT
        checks.append(
            Check(
                f"other share, {phase.phase}",
                Status.PASS if other_ok else Status.FAIL,
                f"{percent(phase.other_ratio)} (limit {OTHER_LIMIT:.0%})",
            )
        )
        unmatched_ok = phase.unmatched_ratio is None or phase.unmatched_ratio <= UNMATCHED_LIMIT
        checks.append(
            Check(
                f"unmatched share, {phase.phase}",
                Status.PASS if unmatched_ok else Status.WARN,
                f"{percent(phase.unmatched_ratio)} of GPU time has no launch (limit {UNMATCHED_LIMIT:.0%})",
            )
        )
        checks.append(Check(f"spill share, {phase.phase}", Status.INFO, percent(phase.spill_ratio)))
    checks.append(_graph_launches(summary))
    checks.append(_last_step(summary))
    return checks


def _graph_launches(summary: Summary) -> Check:
    seen = next((entry for entry in summary.latency.by_name if entry.name == GRAPH_LAUNCH), None)
    if seen is None:
        return Check(
            f"{GRAPH_LAUNCH} seen", Status.WARN, "none; CUDA Graph replays may carry no correlation id"
        )
    detail = f"{seen.count} launches, p50 {us(seen.p50)} us, p99 {us(seen.p99)} us"
    return Check(f"{GRAPH_LAUNCH} seen", Status.INFO, detail)


def _last_step(summary: Summary) -> Check:
    times = [t.step_time for t in summary.step_times]
    if len(times) < 3:
        return Check("last step length", Status.INFO, "fewer than 3 steps")
    ratio = times[-1] / Decimal(statistics.median(times[:-1]))
    low, high = LAST_STEP_BAND
    status = Status.PASS if low <= ratio <= high else Status.WARN
    return Check("last step length", status, f"{ratio:.2f}x the median of the other steps")


def format_report(checks: list[Check]) -> str:
    return "\n".join(f"{c.status.value:<5} {c.name}: {c.detail}" for c in checks) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace")
    parser.add_argument("--prefill-steps", type=int)
    parser.add_argument("--skip-steps", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rules")
    args = parser.parse_args(argv)
    options = Options(
        prefill_steps=args.prefill_steps,
        skip_steps=args.skip_steps,
        device=args.device,
        rules=args.rules,
        show_unclassified=15,
    )
    try:
        summary = summarize(args.trace, options)
    except AnalyzerError as error:
        print(f"error: {error}", file=sys.stderr)
        return error.exit_code
    for warning in summary.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    checks = evaluate(summary)
    sys.stdout.write(format_report(checks))
    if summary.unclassified:
        sys.stdout.write("\ntop unclassified kernels (add rules for these)\n")
        sys.stdout.writelines(f"  {us(duration):>14} us  {name}\n" for name, duration in summary.unclassified)
    return 1 if any(c.status is Status.FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
