from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .loader import load_trace
from .rules import OTHER, Classifier
from .steps import DEFAULT_PREFILL_THRESHOLD, split_steps
from .timeline import StepTime, decompose

PHASE_ORDER = ("prefill", "decode", "unknown")
ZERO = Decimal(0)


@dataclass(frozen=True)
class Options:
    prefill_steps: int | None = None
    prefill_threshold: Decimal = DEFAULT_PREFILL_THRESHOLD
    skip_steps: int = 0
    rules: str | None = None
    no_default_rules: bool = False
    device: int = 0
    show_unclassified: int | None = None


@dataclass(frozen=True)
class PhaseSummary:
    """Per-step averages for one phase, in microseconds."""

    phase: str
    steps: int
    step_time_mean: Decimal
    step_time_max: Decimal
    busy_mean: Decimal
    idle_mean: Decimal
    overlap_mean: Decimal
    bubble_mean: Decimal
    exclusive_mean: dict[str, Decimal]
    inclusive_mean: dict[str, Decimal]
    unmatched_ratio: Decimal | None
    spill_ratio: Decimal | None

    def share(self, category: str) -> Decimal:
        return self.exclusive_mean.get(category, ZERO) / self.step_time_mean if self.step_time_mean else ZERO

    @property
    def other_ratio(self) -> Decimal:
        return self.share(OTHER)


@dataclass(frozen=True)
class Summary:
    trace_path: Path
    options: Options
    device: int
    classifier: Classifier
    phases: list[PhaseSummary]
    step_times: list[StepTime]
    dropped_gpu_us: Decimal
    unclassified: list[tuple[str, Decimal]] | None
    warnings: list[str]

    def phase(self, name: str) -> PhaseSummary | None:
        return next((p for p in self.phases if p.phase == name), None)


def summarize(path: str | Path, options: Options) -> Summary:
    classifier = Classifier.from_file(options.rules, use_defaults=not options.no_default_rules)
    trace = load_trace(path)
    steps = split_steps(
        trace,
        prefill_steps=options.prefill_steps,
        prefill_threshold=options.prefill_threshold,
        skip_steps=options.skip_steps,
    )
    decomposition = decompose(trace, steps, classifier, options.device)
    phases = [
        _phase_summary(name, group, classifier)
        for name in PHASE_ORDER
        if (group := [t for t in decomposition.steps if t.step.phase == name])
    ]
    warnings = list(trace.warnings)
    if not trace.devices:
        warnings.append("the trace has no GPU events, so every step shows 100% idle (a non-CUDA trace?)")
    elif options.device not in trace.devices:
        warnings.append(f"device {options.device} has no GPU events (trace has devices {trace.devices})")
    unclassified = None
    if options.show_unclassified is not None:
        unclassified = _top_unclassified(trace, classifier, options, steps[0].start, steps[-1].end)
    return Summary(
        trace.path, options, options.device, classifier, phases, decomposition.steps,
        decomposition.dropped_gpu_us, unclassified, warnings,
    )  # fmt: skip


def _phase_summary(name: str, group: list[StepTime], classifier: Classifier) -> PhaseSummary:
    count = len(group)

    def mean(values) -> Decimal:
        return sum(values, ZERO) / count

    def per_category(field: str) -> dict[str, Decimal]:
        totals = {c: mean(getattr(t, field).get(c, ZERO) for t in group) for c in classifier.categories()}
        return {c: v for c, v in totals.items() if v}

    gpu, unmatched = sum((t.gpu_us for t in group), ZERO), sum((t.unmatched_us for t in group), ZERO)
    matched, spill = sum((t.matched_us for t in group), ZERO), sum((t.spill_us for t in group), ZERO)
    return PhaseSummary(
        phase=name,
        steps=count,
        step_time_mean=mean(t.step_time for t in group),
        step_time_max=max(t.step_time for t in group),
        busy_mean=mean(t.busy_union for t in group),
        idle_mean=mean(t.idle for t in group),
        overlap_mean=mean(t.overlap for t in group),
        bubble_mean=mean(t.bubble for t in group),
        exclusive_mean=per_category("exclusive"),
        inclusive_mean=per_category("inclusive"),
        unmatched_ratio=unmatched / gpu if gpu else None,
        spill_ratio=spill / matched if matched else None,
    )


def _top_unclassified(trace, classifier: Classifier, options: Options, low: Decimal, high: Decimal):
    totals: dict[str, Decimal] = {}
    for event in trace.gpu_events:
        if (
            event.device == options.device
            and low <= event.ts < high
            and classifier.classify(event.name, event.cat) == OTHER
        ):
            totals[event.name] = totals.get(event.name, ZERO) + event.dur
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return ranked[: options.show_unclassified]
