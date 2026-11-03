"""Time `summary` on a trace from make_synthetic_trace.py and check the answer, not just the clock.

    uv run python scripts/bench_summary.py out.json --runs 5

Reports wall time and peak resident memory of a fresh process per run (median and range), and fails
if the output differs from the closed-form numbers in out.expected.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
from decimal import Decimal
from pathlib import Path


def _timed_command(command: list[str]) -> tuple[list[str], str]:
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l", *command], "darwin"
    return ["/usr/bin/time", "-v", *command], "linux"


def _parse_cost(stderr: str, flavor: str) -> tuple[float, float]:
    """Return (wall seconds, peak resident MB) from /usr/bin/time output."""
    if flavor == "darwin":
        wall = float(re.search(r"([\d.]+)\s+real", stderr).group(1))
        rss = int(re.search(r"(\d+)\s+maximum resident set size", stderr).group(1)) / 1e6
        return wall, rss
    clock = re.search(r"Elapsed \(wall clock\) time.*: (?:(\d+):)?(\d+):([\d.]+)", stderr)
    hours, minutes, seconds = (clock.group(1) or "0"), clock.group(2), clock.group(3)
    wall = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return wall, int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr).group(1)) / 1e3


def _check(document: dict, expected: dict) -> list[str]:
    """Compare every number the generator can predict; an empty list means the answer is right."""
    steps = document["steps"]
    if len(steps) != expected["steps"] + 1:
        return [f"expected {expected['steps'] + 1} steps, got {len(steps)}"]
    step_time = Decimal(expected["step_time_us"])
    idle = Decimal(expected["idle_us"])
    wanted = {
        "step_time": step_time,
        "idle": idle,
        "busy_union": step_time - idle,
        "overlap": Decimal(expected["overlap_us"]),
    }
    problems = []
    for step in steps[: expected["steps"]]:
        for key, value in wanted.items():
            if Decimal(step[key]) != value:
                problems.append(f"step {step['index']}: {key} {step[key]}, expected {value}")
        for key in ("exclusive", "inclusive"):
            got = {c: Decimal(v) for c, v in step[key].items()}
            want = {c: Decimal(v) for c, v in expected[f"{key}_us"].items()}
            if got != want:
                problems.append(f"step {step['index']}: {key} {step[key]}")
    trailing = steps[-1]  # the closing schedule event: a 1 us step with no GPU work
    if (Decimal(trailing["step_time"]), Decimal(trailing["idle"]), trailing["exclusive"]) != (1, 1, {}):
        problems.append(
            f"trailing step: {trailing['step_time']} us, idle {trailing['idle']}, {trailing['exclusive']}"
        )
    return problems[:5]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)
    trace = Path(args.trace)
    expected = json.loads(trace.with_suffix(".expected.json").read_text())
    command = [sys.executable, "-m", "deepinfer_trace_analyzer", "summary", str(trace)]
    command += ["--prefill-steps", "0", "--format", "json"]
    timed, flavor = _timed_command(command)

    walls, rsses = [], []
    for run in range(args.runs):
        result = subprocess.run(timed, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"run {run}: exit {result.returncode}\n{result.stderr[-800:]}", file=sys.stderr)
            return 1
        problems = _check(json.loads(result.stdout), expected)
        if problems:
            print(
                f"run {run}: output differs from the closed-form answer:\n  " + "\n  ".join(problems),
                file=sys.stderr,
            )
            return 1
        wall, rss = _parse_cost(result.stderr, flavor)
        walls.append(wall)
        rsses.append(rss)
        print(f"run {run}: {wall:.1f} s, {rss:.0f} MB peak, output verified", file=sys.stderr)

    size_mb = trace.stat().st_size / 1e6
    print(
        f"{size_mb:.0f} MB trace, {expected['events']} events, {args.runs} runs: "
        f"time median {statistics.median(walls):.1f} s (range {min(walls):.1f} to {max(walls):.1f}), "
        f"peak memory median {statistics.median(rsses):.0f} MB (range {min(rsses):.0f} to {max(rsses):.0f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
