from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal

from .errors import AnalysisError
from .model import Trace

LAUNCH_FORWARD = re.compile(r"^deepinfer:launch_forward_N(\d+)_B(\d+)$")
DEFAULT_PREFILL_THRESHOLD = Decimal(16)


@dataclass(frozen=True, slots=True)
class Step:
    index: int
    start: Decimal
    end: Decimal
    phase: str

    @property
    def duration(self) -> Decimal:
        return self.end - self.start


def split_steps(
    trace: Trace,
    *,
    prefill_steps: int | None = None,
    prefill_threshold: Decimal = DEFAULT_PREFILL_THRESHOLD,
    skip_steps: int = 0,
) -> list[Step]:
    """Cut the trace into windows [schedule[k], schedule[k+1]) and label each prefill/decode/unknown."""
    starts = trace.schedule_starts
    if prefill_steps is not None and prefill_steps > len(starts):
        raise AnalysisError(f"--prefill-steps {prefill_steps} exceeds the {len(starts)} steps in the trace")
    if skip_steps >= len(starts):
        raise AnalysisError(f"--skip-steps {skip_steps} leaves none of the {len(starts)} steps")

    ends = [*starts[1:], max(starts[-1], _last_event_end(trace))]
    shapes = _launch_shapes(trace, starts)
    steps = []
    for index in range(skip_steps, len(starts)):
        if prefill_steps is not None:
            phase = "prefill" if index < prefill_steps else "decode"
        else:
            phase = _phase_from_shape(shapes.get(index), Decimal(prefill_threshold))
        steps.append(Step(index, starts[index], ends[index], phase))
    return steps


def _last_event_end(trace: Trace) -> Decimal:
    ends = [e.ts + e.dur for e in trace.cpu_events]
    ends.extend(e.end for e in trace.gpu_events)
    return max(ends)


def _launch_shapes(trace: Trace, starts: list[Decimal]) -> dict[int, tuple[int, int]]:
    """First launch_forward (requests, tokens) seen inside each step window."""
    shapes: dict[int, tuple[int, int]] = {}
    for event in sorted(trace.cpu_events, key=lambda e: e.ts):
        match = LAUNCH_FORWARD.match(event.name)
        if match is None:
            continue
        index = bisect_right(starts, event.ts) - 1
        if index >= 0 and index not in shapes:
            shapes[index] = (int(match[1]), int(match[2]))
    return shapes


def _phase_from_shape(shape: tuple[int, int] | None, threshold: Decimal) -> str:
    if shape is None or shape[0] == 0:
        return "unknown"
    requests, tokens = shape
    return "prefill" if Decimal(tokens) / requests >= threshold else "decode"
