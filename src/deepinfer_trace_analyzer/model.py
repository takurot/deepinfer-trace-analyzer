from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

GPU_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
MEMORY_CATEGORIES = frozenset({"gpu_memcpy", "gpu_memset"})
SCHEDULE_NAME = "deepinfer:schedule"


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
