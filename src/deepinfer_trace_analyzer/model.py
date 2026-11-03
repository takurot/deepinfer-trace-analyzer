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
    warnings: list[str] = field(default_factory=list)

    @property
    def schedule_starts(self) -> list[Decimal]:
        return sorted(e.ts for e in self.cpu_events if e.name == SCHEDULE_NAME)

    @property
    def devices(self) -> list[int]:
        return sorted({e.device for e in self.gpu_events})
