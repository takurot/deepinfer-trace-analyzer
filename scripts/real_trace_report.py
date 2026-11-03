"""Pass/fail report for a trace taken on real hardware (issue #11, SPEC section 6 criterion 2 and section 8).

    uv run python scripts/real_trace_report.py trace.json --prefill-steps 2

Exit status is 0 if no hard check fails, 1 if one does (too much time in `other`, no GPU events,
no decode steps), and 2 if the trace cannot be read or analyzed. Warnings and
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
from deepinfer_trace_analyzer.model import OTHER_SHARE_LIMIT, UNMATCHED_SHARE_LIMIT
from deepinfer_trace_analyzer.summary import Options, Summary, summarize

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
    checks = [_gpu_events(summary), _decode_phase(summary)]
    for phase in summary.phases:
        other_ok = phase.other_ratio <= OTHER_SHARE_LIMIT
        checks.append(
            Check(
                f"other share, {phase.phase}",
                Status.PASS if other_ok else Status.FAIL,
                f"{percent(phase.other_ratio)} (limit {OTHER_SHARE_LIMIT:.0%})",
            )
        )
        unmatched_ok = phase.unmatched_ratio is None or phase.unmatched_ratio <= UNMATCHED_SHARE_LIMIT
        checks.append(
            Check(
                f"unmatched share, {phase.phase}",
                Status.PASS if unmatched_ok else Status.WARN,
                f"{percent(phase.unmatched_ratio)} of GPU time has no launch "
                f"(limit {UNMATCHED_SHARE_LIMIT:.0%})",
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


def _gpu_events(summary: Summary) -> Check:
    if not summary.devices:
        return Check("GPU events present", Status.FAIL, "none; a trace without GPU work proves nothing")
    return Check("GPU events present", Status.PASS, f"devices {summary.devices}")


def _decode_phase(summary: Summary) -> Check:
    if any(phase.phase == "decode" for phase in summary.phases):
        return Check("decode phase present", Status.PASS, "")
    if summary.options.prefill_steps is not None:
        return Check(
            "decode phase present",
            Status.FAIL,
            f"no steps after --prefill-steps {summary.options.prefill_steps}",
        )
    return Check(
        "decode phase present", Status.WARN, "no step was classified as decode; pass --prefill-steps N"
    )


def _last_step(summary: Summary) -> Check:
    """Compare the last step with the median of the earlier steps of its own phase."""
    last = summary.step_times[-1]
    others = [t.step_time for t in summary.step_times[:-1] if t.step.phase == last.step.phase]
    if len(others) < 2:
        return Check("last step length", Status.INFO, f"fewer than 3 steps in {last.step.phase}")
    ratio = last.step_time / Decimal(statistics.median(others))
    low, high = LAST_STEP_BAND
    status = Status.PASS if low <= ratio <= high else Status.WARN
    return Check(
        "last step length", status, f"{ratio:.2f}x the median of the earlier {last.step.phase} steps"
    )


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
        return 2
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
