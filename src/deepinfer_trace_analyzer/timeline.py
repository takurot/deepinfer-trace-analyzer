from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import NamedTuple

from .model import Trace
from .rules import Classifier
from .steps import Step

ZERO = Decimal(0)
# Wide enough that sums of nanosecond-resolution timestamps stay exact.
PRECISION = 60


class _Interval(NamedTuple):
    start: Decimal
    end: Decimal
    category: str
    launch_step: int | None


@dataclass(frozen=True, slots=True)
class StepTime:
    """GPU time inside one step window of one device, in microseconds."""

    step: Step
    busy_union: Decimal
    exclusive: dict[str, Decimal]
    inclusive: dict[str, Decimal]
    bubble: Decimal
    gpu_us: Decimal
    matched_us: Decimal
    spill_us: Decimal
    unmatched_us: Decimal

    @property
    def step_time(self) -> Decimal:
        return self.step.duration

    @property
    def idle(self) -> Decimal:
        return self.step_time - self.busy_union

    @property
    def overlap(self) -> Decimal:
        return sum(self.inclusive.values(), ZERO) - self.busy_union


@dataclass(frozen=True, slots=True)
class Decomposition:
    device: int
    steps: list[StepTime]
    dropped_gpu_us: Decimal


def decompose(trace: Trace, steps: list[Step], classifier: Classifier, device: int) -> Decomposition:
    """Cut a device's GPU activity at step boundaries and split each window's busy time by category.

    Exclusive time hands every instant to the highest-priority category running then, so
    sum(exclusive) + idle == step_time. Inclusive time sums durations and counts overlap twice.
    """
    with localcontext() as context:
        context.prec = PRECISION
        intervals = _intervals(trace, classifier, device)
        starts = [step.start for step in steps]
        pieces: list[list[_Interval]] = [[] for _ in steps]
        for interval in intervals:
            for index, start, end in _slices(interval.start, interval.end, starts, steps):
                inside = start == interval.start and end == interval.end
                pieces[index].append(interval if inside else interval._replace(start=start, end=end))
        bubbles = _bubble_per_window(trace, starts, steps)
        times = [
            _step_time(step, window, bubble, classifier)
            for step, window, bubble in zip(steps, pieces, bubbles, strict=True)
        ]
        return Decomposition(device, times, _dropped(intervals, trace.schedule_starts[0], steps[-1].end))


def _intervals(trace: Trace, classifier: Classifier, device: int) -> list[_Interval]:
    """The only place that reads GpuEvent fields; everything after works on these tuples."""
    schedule_starts = trace.schedule_starts
    launches = trace.launches
    result = []
    for event in trace.gpu_events:
        if event.device != device or event.dur <= 0:
            continue
        launch = launches.get(event.correlation) if event.correlation is not None else None
        launch_step = None if launch is None else bisect_right(schedule_starts, launch.ts) - 1
        result.append(_Interval(event.ts, event.end, classifier.classify(event.name, event.cat), launch_step))
    result.sort(key=lambda interval: interval.start)
    return result


def _slices(
    start: Decimal, end: Decimal, starts: list[Decimal], steps: list[Step]
) -> Iterator[tuple[int, Decimal, Decimal]]:
    """The parts of [start, end) that fall in each (contiguous) step window."""
    index = max(bisect_right(starts, start) - 1, 0)
    while index < len(steps) and steps[index].start < end:
        low, high = max(start, steps[index].start), min(end, steps[index].end)
        if high > low:
            yield index, low, high
        index += 1


def _bubble_per_window(trace: Trace, starts: list[Decimal], steps: list[Step]) -> list[Decimal]:
    totals = [ZERO] * len(steps)
    for event in trace.bubbles:
        for index, low, high in _slices(event.ts, event.ts + event.dur, starts, steps):
            totals[index] += high - low
    return totals


def _step_time(step: Step, pieces: list[_Interval], bubble: Decimal, classifier: Classifier) -> StepTime:
    inclusive: dict[str, Decimal] = {}
    matched = spill = unmatched = total = ZERO
    for piece in pieces:
        length = piece.end - piece.start
        inclusive[piece.category] = inclusive.get(piece.category, ZERO) + length
        total += length
        if piece.launch_step is None:
            unmatched += length
        else:
            matched += length
            if piece.launch_step < step.index:
                spill += length
    exclusive = _exclusive(pieces, classifier)
    return StepTime(
        step, sum(exclusive.values(), ZERO), exclusive, inclusive, bubble, total, matched, spill, unmatched
    )


def _exclusive(pieces: list[_Interval], classifier: Classifier) -> dict[str, Decimal]:
    # Sorting on time alone is enough: segments of zero length add nothing, so the order of an
    # end and a start at the same instant does not matter.
    edges = sorted(
        [(p.start, 1, p.category) for p in pieces] + [(p.end, -1, p.category) for p in pieces],
        key=lambda edge: edge[0],
    )
    rank = {category: classifier.priority(category) for category in {p.category for p in pieces}}
    active: dict[str, int] = {}
    exclusive: dict[str, Decimal] = {}
    previous = None
    for time, delta, category in edges:
        if active and previous is not None and time > previous:
            winner = min(active, key=rank.__getitem__)
            exclusive[winner] = exclusive.get(winner, ZERO) + (time - previous)
        previous = time
        count = active.get(category, 0) + delta
        if count:
            active[category] = count
        else:
            del active[category]
    return exclusive


def _dropped(intervals: list[_Interval], low: Decimal, high: Decimal) -> Decimal:
    dropped = ZERO
    for interval in intervals:
        inside = max(min(interval.end, high) - max(interval.start, low), ZERO)
        dropped += interval.end - interval.start - inside
    return dropped
