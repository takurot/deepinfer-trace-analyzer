from __future__ import annotations

import json
from dataclasses import fields
from decimal import Decimal

from .fmt import dec, percent, us
from .model import IDLE, ZERO
from .summary import PhaseSummary, Summary
from .timeline import StepTime

SCHEMA_VERSION = 1


def render_summary(summary: Summary, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(_document(summary), indent=2) + "\n"
    return _markdown(summary) if fmt == "markdown" else _table(summary)


def _heading(phase: PhaseSummary) -> str:
    plural = "step" if phase.steps == 1 else "steps"
    mean, longest = us(phase.step_time_mean), us(phase.step_time_max)
    return f"{phase.phase}: {phase.steps} {plural}, step time mean {mean} us, max {longest} us"


def _rows(phase: PhaseSummary) -> list[tuple[str, str, str, str]]:
    categories = [*phase.exclusive_mean, *(c for c in phase.inclusive_mean if c not in phase.exclusive_mean)]
    rows = [
        (
            c,
            us(phase.exclusive_mean.get(c, ZERO)),
            percent(phase.share(c)),
            us(phase.inclusive_mean.get(c, ZERO)),
        )
        for c in categories
    ]
    idle_share = phase.idle_mean / phase.step_time_mean if phase.step_time_mean else ZERO
    rows.append((IDLE, us(phase.idle_mean), percent(idle_share), ""))
    return rows


def _diagnostics(phase: PhaseSummary) -> str:
    return (
        f"overlap {us(phase.overlap_mean)} us, bubble {us(phase.bubble_mean)} us, "
        f"other {percent(phase.other_ratio)}, unmatched {percent(phase.unmatched_ratio)}, "
        f"spill {percent(phase.spill_ratio)}"
    )


HEADER = ("category", "exclusive us/step", "% of step", "inclusive us/step")


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
        lines += [f"  {us(duration):>14} us  {name}" for name, duration in summary.unclassified] or ["  none"]
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
        lines += [f"- `{name}` {us(duration)} us" for name, duration in summary.unclassified] or ["- none"]
    return "\n".join(lines).rstrip("\n") + "\n"


def _footer(summary: Summary) -> list[str]:
    if summary.dropped_gpu_us:
        return [f"GPU time before the first step: {us(summary.dropped_gpu_us)} us (not counted)", ""]
    return []


def _document(summary: Summary) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": "us",
        "trace": str(summary.trace_path),
        "device": summary.device,
        "options": options_document(summary.options),
        "rules": rules_document(summary.classifier.rules()),
        "phases": [_phase_document(p) for p in summary.phases],
        "steps": [_step_document(t) for t in summary.step_times],
        "dropped_gpu_us": dec(summary.dropped_gpu_us),
        "unclassified": None
        if summary.unclassified is None
        else [{"name": n, "duration_us": dec(d)} for n, d in summary.unclassified],
        "warnings": summary.warnings,
    }


def options_document(options) -> dict:
    return {f.name: _plain(getattr(options, f.name)) for f in fields(options)}


def rules_document(rules: list[tuple[str, str]]) -> list[dict]:
    return [{"category": category, "pattern": pattern} for category, pattern in rules]


def _plain(value):
    return dec(value) if isinstance(value, Decimal) else value


def _phase_document(phase: PhaseSummary) -> dict:
    return {
        "phase": phase.phase,
        "steps": phase.steps,
        "step_time_mean": dec(phase.step_time_mean),
        "step_time_max": dec(phase.step_time_max),
        "busy_mean": dec(phase.busy_mean),
        "idle_mean": dec(phase.idle_mean),
        "overlap_mean": dec(phase.overlap_mean),
        "bubble_mean": dec(phase.bubble_mean),
        "exclusive_mean": {c: dec(v) for c, v in phase.exclusive_mean.items()},
        "inclusive_mean": {c: dec(v) for c, v in phase.inclusive_mean.items()},
        "other_ratio": dec(phase.other_ratio),
        "unmatched_ratio": None if phase.unmatched_ratio is None else dec(phase.unmatched_ratio),
        "spill_ratio": None if phase.spill_ratio is None else dec(phase.spill_ratio),
    }


def _step_document(step_time: StepTime) -> dict:
    step = step_time.step
    return {
        "index": step.index,
        "phase": step.phase,
        "start": dec(step.start),
        "end": dec(step.end),
        "step_time": dec(step_time.step_time),
        "busy_union": dec(step_time.busy_union),
        "idle": dec(step_time.idle),
        "overlap": dec(step_time.overlap),
        "exclusive": {c: dec(v) for c, v in step_time.exclusive.items()},
        "inclusive": {c: dec(v) for c, v in step_time.inclusive.items()},
        "bubble": dec(step_time.bubble),
        "gpu_us": dec(step_time.gpu_us),
        "matched_us": dec(step_time.matched_us),
        "spill_us": dec(step_time.spill_us),
        "unmatched_us": dec(step_time.unmatched_us),
    }
