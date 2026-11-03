from __future__ import annotations

import json
from dataclasses import fields
from decimal import Decimal

from .diff import Diff, PhaseDiff, Row
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
    mean, longest = _us(phase.step_time_mean), _us(phase.step_time_max)
    return f"{phase.phase}: {phase.steps} {plural}, step time mean {mean} us, max {longest} us"


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
        f"overlap {_us(phase.overlap_mean)} us, bubble {_us(phase.bubble_mean)} us, "
        f"other {_percent(phase.other_ratio)}, unmatched {_percent(phase.unmatched_ratio)}, "
        f"spill {_percent(phase.spill_ratio)}"
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


def render_diff(diff: Diff, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(_diff_document(diff), indent=2) + "\n"
    return _diff_markdown(diff) if fmt == "markdown" else _diff_table(diff)


def _signed(value: Decimal) -> str:
    return f"{value:+.3f}"


def _diff_heading(phase: PhaseDiff) -> str:
    return (
        f"{phase.phase}: {phase.steps_before} vs {phase.steps_after} steps, step time "
        f"{_us(phase.step_time_before)} -> {_us(phase.step_time_after)} us ({_signed(phase.step_time_delta)})"
    )


DIFF_HEADER = ("category", "before us", "after us", "delta us", "share before", "share after", "delta pt")


def _diff_cells(row: Row) -> tuple[str, ...]:
    shares = (
        (_percent(row.share_before), _percent(row.share_after), _signed(row.share_delta_pt))
        if row.share_delta_pt is not None
        else ("", "", "")
    )
    return (row.name, _us(row.before), _us(row.after), _signed(row.delta), *shares)


def _overlap_line(phase: PhaseDiff) -> str:
    overlap = phase.overlap
    return f"overlap {_us(overlap.before)} -> {_us(overlap.after)} us ({_signed(overlap.delta)})"


def _residual_line(phase: PhaseDiff) -> str:
    change, residual = _signed(phase.step_time_delta), _us(phase.residual)
    return f"step time change {change} us = sum of rows, residual {residual} us"


def _diff_table(diff: Diff) -> str:
    lines = [f"before: {diff.before_path}", f"after: {diff.after_path}", f"device: {diff.device}", ""]
    for phase in diff.phases:
        lines.append(_diff_heading(phase))
        sections = [("exclusive", phase.rows), ("inclusive", phase.inclusive_rows)]
        for title, source in sections:
            cells = [DIFF_HEADER, *(_diff_cells(row) for row in source)]
            widths = [max(len(c[i]) for c in cells) for i in range(len(DIFF_HEADER))]
            lines.append(f"  {title}")
            for cell in cells:
                head, rest = cell[0].ljust(widths[0]), (cell[i].rjust(widths[i]) for i in range(1, len(cell)))
                lines.append("    " + head + "  " + "  ".join(rest))
        lines += ["  " + _overlap_line(phase), "  " + _residual_line(phase), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _diff_markdown(diff: Diff) -> str:
    lines = [f"**before** `{diff.before_path}`, **after** `{diff.after_path}`, **device** {diff.device}", ""]
    for phase in diff.phases:
        lines += [f"## {_diff_heading(phase)}", ""]
        for title, source in (("Exclusive", phase.rows), ("Inclusive", phase.inclusive_rows)):
            lines += [f"### {title}", "", "| " + " | ".join(DIFF_HEADER) + " |", "|---|" + "--:|" * 6]
            lines += ["| " + " | ".join(_diff_cells(row)) + " |" for row in source]
            lines.append("")
        lines += [_overlap_line(phase), "", _residual_line(phase), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _row_document(row: Row) -> dict:
    document = {"name": row.name, "before": str(row.before), "after": str(row.after), "delta": str(row.delta)}
    if row.share_delta_pt is not None:
        document |= {
            "share_before": str(row.share_before),
            "share_after": str(row.share_after),
            "share_delta_pt": str(row.share_delta_pt),
        }
    return document


def _diff_document(diff: Diff) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "before": str(diff.before_path),
        "after": str(diff.after_path),
        "device": diff.device,
        "phases": [
            {
                "phase": p.phase,
                "steps_before": p.steps_before,
                "steps_after": p.steps_after,
                "step_time_before": str(p.step_time_before),
                "step_time_after": str(p.step_time_after),
                "step_time_delta": str(p.step_time_delta),
                "rows": [_row_document(r) for r in p.rows],
                "inclusive_rows": [_row_document(r) for r in p.inclusive_rows],
                "overlap": _row_document(p.overlap),
                "residual": str(p.residual),
            }
            for p in diff.phases
        ],
        "warnings": diff.warnings,
    }
