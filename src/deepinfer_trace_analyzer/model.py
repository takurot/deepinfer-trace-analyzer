from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from functools import cached_property
from pathlib import Path

GPU_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
MEMORY_CATEGORIES = frozenset({"gpu_memcpy", "gpu_memset"})
ZERO = Decimal(0)
# Shares above these make a trace untrustworthy: SPEC section 6 (criterion 2) and FR-7.
OTHER_SHARE_LIMIT = Decimal("0.10")
UNMATCHED_SHARE_LIMIT = Decimal("0.05")
IDLE = "idle"
SCHEDULE_NAME = "deepinfer:schedule"
BUBBLE_NAME = "deepinfer:bubble"


@dataclass(frozen=True, slots=True)
class CpuEvent:
    name: str
    ts: Decimal
    dur: Decimal
    correlation: int | None


@dataclass(frozen=True, slots=True)
class GpuEvent:
    name: str
    cat: str
    device: int
    ts: Decimal
    dur: Decimal
    correlation: int | None

    @property
    def end(self) -> Decimal:
        return self.ts + self.dur


@dataclass
class Trace:
    path: Path
    cpu_events: list[CpuEvent]
    gpu_events: list[GpuEvent]
    schedule_starts: list[Decimal]
    devices: list[int]
    warnings: list[str] = field(default_factory=list)

    @cached_property
    def launches(self) -> dict[int, CpuEvent]:
        """The earliest CPU event per correlation id: the launch that queued the GPU work."""
        earliest: dict[int, CpuEvent] = {}
        for event in self.cpu_events:
            known = earliest.get(event.correlation)
            if event.correlation is not None and (known is None or event.ts < known.ts):
                earliest[event.correlation] = event
        return earliest

    @cached_property
    def bubbles(self) -> list[CpuEvent]:
        return [event for event in self.cpu_events if event.name == BUBBLE_NAME]
