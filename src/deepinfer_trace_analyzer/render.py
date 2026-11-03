from __future__ import annotations

import json
from dataclasses import fields
from decimal import Decimal

from .summary import PhaseSummary, Summary
from .timeline import StepTime

SCHEMA_VERSION = 1
IDLE = "idle"


def render_summary(summary: Summary, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(_document(summary), indent=2) + "\n"
    return _markdown(summary) if fmt == "markdown" else _table(summary)


def _us(value: Decimal) -> str:
    return f"{value:.3f}"


def _percent(value: Decimal | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _heading(phase: PhaseSummary) -> str:
    plural = "step" if phase.steps == 1 else "steps"
    return f"{phase.phase}: {phase.steps} {plural}, step time mean {_us(phase.step_time_mean)} us, max {_us(phase.step_time_max)} us"


def _rows(phase: PhaseSummary) -> list[tuple[str, str, str, str]]:
    categories = [*phase.exclusive_mean, *(c for c in phase.inclusive_mean if c not in phase.exclusive_mean)]
    rows = [
        (
            c,
            _us(phase.exclusive_mean.get(c, Decimal(0))),
            _percent(phase.share(c)),
            _us(phase.inclusive_mean.get(c, Decimal(0))),
        )
        for c in categories
    ]
    idle_share = phase.idle_mean / phase.step_time_mean if phase.step_time_mean else Decimal(0)
    rows.append((IDLE, _us(phase.idle_mean), _percent(idle_share), ""))
    return rows


def _diagnostics(phase: PhaseSummary) -> str:
    return (
        f"overlap {_us(phase.overlap_mean)} us, bubble {_us(phase.bubble_mean)} us, other {_percent(phase.other_ratio)}, "
        f"unmatched {_percent(phase.unmatched_ratio)}, spill {_percent(phase.spill_ratio)}"
    )


HEADER = ("category", "exclusive us/step", "share", "inclusive us/step")


def _table(summary: Summary) -> str:
    lines = [f"trace: {summary.trace_path}", f"device: {summary.device}", ""]
    for phase in summary.phases:
        rows = [HEADER, *_rows(phase)]
        widths = [max(len(row[i]) for row in rows) for i in range(4)]
        lines.append(_heading(phase))
        for row in rows:
            lines.append(
                "  " + row[0].ljust(widths[0]) + "  " + "  ".join(row[i].rjust(widths[i]) for i in (1, 2, 3))
            )
        lines += ["  " + _diagnostics(phase), ""]
    lines += _footer(summary)
    if summary.unclassified is not None:
        lines.append(f"unclassified kernels (top {summary.options.show_unclassified} by total duration)")
        lines += [f"  {_us(duration):>14} us  {name}" for name, duration in summary.unclassified] or [
            "  none"
        ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _markdown(summary: Summary) -> str:
    lines = [f"**trace** `{summary.trace_path}`, **device** {summary.device}", ""]
    for phase in summary.phases:
        lines += [f"## {_heading(phase)}", "", "| " + " | ".join(HEADER) + " |", "|---|--:|--:|--:|"]
        lines += [f"| {c} | {e} | {s} | {i} |" for c, e, s, i in _rows(phase)]
        lines += ["", _diagnostics(phase), ""]
    lines += _footer(summary)
    if summary.unclassified is not None:
        lines += [
            "",
            f"### Unclassified kernels (top {summary.options.show_unclassified} by total duration)",
            "",
        ]
        lines += [f"- `{name}` {_us(duration)} us" for name, duration in summary.unclassified] or ["- none"]
    return "\n".join(lines).rstrip("\n") + "\n"


def _footer(summary: Summary) -> list[str]:
    if summary.dropped_gpu_us:
        return [f"GPU time before the first step: {_us(summary.dropped_gpu_us)} us (not counted)", ""]
    return []


def _document(summary: Summary) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "trace": str(summary.trace_path),
        "device": summary.device,
        "options": {f.name: _plain(getattr(summary.options, f.name)) for f in fields(summary.options)},
        "rules": [{"category": c, "pattern": p} for c, p in summary.classifier.rules()],
        "phases": [_phase_document(p) for p in summary.phases],
        "steps": [_step_document(t) for t in summary.step_times],
        "dropped_gpu_us": str(summary.dropped_gpu_us),
        "unclassified": None
        if summary.unclassified is None
        else [{"name": n, "duration_us": str(d)} for n, d in summary.unclassified],
        "warnings": summary.warnings,
    }


def _plain(value):
    return str(value) if isinstance(value, Decimal) else value


def _phase_document(phase: PhaseSummary) -> dict:
    return {
        "phase": phase.phase,
        "steps": phase.steps,
        "step_time_mean": str(phase.step_time_mean),
        "step_time_max": str(phase.step_time_max),
        "busy_mean": str(phase.busy_mean),
        "idle_mean": str(phase.idle_mean),
        "overlap_mean": str(phase.overlap_mean),
        "bubble_mean": str(phase.bubble_mean),
        "exclusive_mean": {c: str(v) for c, v in phase.exclusive_mean.items()},
        "inclusive_mean": {c: str(v) for c, v in phase.inclusive_mean.items()},
        "other_ratio": str(phase.other_ratio),
        "unmatched_ratio": None if phase.unmatched_ratio is None else str(phase.unmatched_ratio),
        "spill_ratio": None if phase.spill_ratio is None else str(phase.spill_ratio),
    }


def _step_document(step_time: StepTime) -> dict:
    step = step_time.step
    return {
        "index": step.index,
        "phase": step.phase,
        "start": str(step.start),
        "end": str(step.end),
        "step_time": str(step_time.step_time),
        "busy_union": str(step_time.busy_union),
        "idle": str(step_time.idle),
        "overlap": str(step_time.overlap),
        "exclusive": {c: str(v) for c, v in step_time.exclusive.items()},
        "inclusive": {c: str(v) for c, v in step_time.inclusive.items()},
        "bubble": str(step_time.bubble),
        "gpu_us": str(step_time.gpu_us),
        "matched_us": str(step_time.matched_us),
        "spill_us": str(step_time.spill_us),
        "unmatched_us": str(step_time.unmatched_us),
    }
