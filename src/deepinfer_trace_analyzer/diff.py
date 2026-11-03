from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .model import IDLE, OTHER_SHARE_LIMIT, UNMATCHED_SHARE_LIMIT, ZERO
from .summary import PHASE_ORDER, Options, PhaseSummary, Summary

# Means are rounded at 28 digits, so a true zero can come out as ~1e-26.
RESIDUAL_NOISE = Decimal("1E-12")


@dataclass(frozen=True)
class Row:
    name: str
    before: Decimal
    after: Decimal
    share_before: Decimal | None = None
    share_after: Decimal | None = None

    @property
    def delta(self) -> Decimal:
        return self.after - self.before

    @property
    def share_delta_pt(self) -> Decimal | None:
        if self.share_before is None or self.share_after is None:
            return None
        return (self.share_after - self.share_before) * 100


@dataclass(frozen=True)
class PhaseDiff:
    phase: str
    steps_before: int
    steps_after: int
    step_time_before: Decimal
    step_time_after: Decimal
    rows: list[Row]
    inclusive_rows: list[Row]
    overlap: Row

    @property
    def step_time_delta(self) -> Decimal:
        return self.step_time_after - self.step_time_before

    @property
    def residual(self) -> Decimal:
        """Step-time change that the exclusive rows and idle do not account for. Always zero."""
        residual = self.step_time_delta - sum((row.delta for row in self.rows), ZERO)
        return residual if abs(residual) > RESIDUAL_NOISE else ZERO


@dataclass(frozen=True)
class Diff:
    before_path: Path
    after_path: Path
    device: int
    options: Options
    rules: list[tuple[str, str]]
    phases: list[PhaseDiff]
    warnings: list[str]


def compare(before: Summary, after: Summary) -> Diff:
    warnings = [f"before: {w}" for w in before.warnings] + [f"after: {w}" for w in after.warnings]
    if len(before.devices) != len(after.devices):
        warnings.append(f"device count differs: {len(before.devices)} before, {len(after.devices)} after")
    phases = []
    for name in PHASE_ORDER:
        old, new = before.phase(name), after.phase(name)
        if old is None and new is None:
            continue
        if old is None or new is None:
            warnings.append(
                f"phase {name} exists only in {'AFTER' if old is None else 'BEFORE'}, so it is not compared"
            )
            continue
        warnings += _phase_warnings(old, new)
        phases.append(_phase_diff(old, new))
    rules = before.classifier.rules()
    return Diff(before.trace_path, after.trace_path, before.device, before.options, rules, phases, warnings)


def _phase_warnings(old: PhaseSummary, new: PhaseSummary) -> list[str]:
    found = []
    if old.steps != new.steps:
        found.append(f"{old.phase}: {old.steps} steps before, {new.steps} after (values are per-step means)")
    for side, phase in (("before", old), ("after", new)):
        if phase.unmatched_ratio is not None and phase.unmatched_ratio > UNMATCHED_SHARE_LIMIT:
            found.append(
                f"{side} {phase.phase}: {phase.unmatched_ratio:.0%} of GPU time is unmatched (no launch)"
            )
        if phase.other_ratio > OTHER_SHARE_LIMIT:
            found.append(
                f"{side} {phase.phase}: {phase.other_ratio:.0%} of step time is 'other'; "
                "try --show-unclassified or --rules"
            )
    return found


def _phase_diff(old: PhaseSummary, new: PhaseSummary) -> PhaseDiff:
    categories = sorted({*old.exclusive_mean, *new.exclusive_mean})
    rows = [
        Row(c, old.exclusive_mean.get(c, ZERO), new.exclusive_mean.get(c, ZERO), old.share(c), new.share(c))
        for c in categories
    ]
    rows.append(
        Row(IDLE, old.idle_mean, new.idle_mean, _share(old.idle_mean, old), _share(new.idle_mean, new))
    )
    inclusive = [
        Row(c, old.inclusive_mean.get(c, ZERO), new.inclusive_mean.get(c, ZERO))
        for c in sorted({*old.inclusive_mean, *new.inclusive_mean})
    ]
    return PhaseDiff(
        phase=old.phase,
        steps_before=old.steps,
        steps_after=new.steps,
        step_time_before=old.step_time_mean,
        step_time_after=new.step_time_mean,
        rows=_by_change(rows),
        inclusive_rows=_by_change(inclusive),
        overlap=Row("overlap", old.overlap_mean, new.overlap_mean),
    )


def _share(value: Decimal, phase: PhaseSummary) -> Decimal:
    return value / phase.step_time_mean if phase.step_time_mean else ZERO


def _by_change(rows: list[Row]) -> list[Row]:
    return sorted(rows, key=lambda row: (-abs(row.delta), row.name))
