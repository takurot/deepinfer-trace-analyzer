from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .model import Trace


@dataclass(frozen=True, slots=True)
class LaunchLatency:
    name: str
    count: int
    p50: Decimal
    p99: Decimal


@dataclass(frozen=True, slots=True)
class LatencyReport:
    """Delay from a CPU launch to the first GPU event it queued, in microseconds."""

    by_name: list[LaunchLatency]
    matched_events: int
    total_events: int

    @property
    def matched_fraction(self) -> Decimal | None:
        return Decimal(self.matched_events) / self.total_events if self.total_events else None


def launch_latency(trace: Trace, device: int) -> LatencyReport:
    """One sample per correlation id, so a CUDA graph launch counts once however many kernels it replays."""
    first_start: dict[int, Decimal] = {}
    total = matched = 0
    for event in trace.gpu_events:
        if event.device != device:
            continue
        total += 1
        if event.correlation is None or event.correlation not in trace.launches:
            continue
        matched += 1
        known = first_start.get(event.correlation)
        if known is None or event.ts < known:
            first_start[event.correlation] = event.ts

    samples: dict[str, list[Decimal]] = {}
    for correlation, start in first_start.items():
        launch = trace.launches[correlation]
        samples.setdefault(launch.name, []).append(start - launch.ts)
    by_name = [
        LaunchLatency(name, len(values), _nearest_rank(values, 50), _nearest_rank(values, 99))
        for name, values in sorted(samples.items())
    ]
    return LatencyReport(by_name, matched, total)


def _nearest_rank(values: list[Decimal], percent: int) -> Decimal:
    ordered = sorted(values)
    rank = -(-percent * len(ordered) // 100)
    return ordered[max(rank, 1) - 1]
