from __future__ import annotations

import json

from .diff import Diff, PhaseDiff, Row
from .fmt import SCHEMA_VERSION, dec, options_document, percent, rules_document, signed, us
from .model import IDLE

EXCLUSIVE_HEADER = (
    "category",
    "before us",
    "after us",
    "delta us",
    "% of step before",
    "% of step after",
    "delta pt",
)
INCLUSIVE_HEADER = ("category", "before us", "after us", "delta us")
LEGEND = (
    "exclusive: time given to one category at a time, so the rows add up to the step time.",
    "inclusive: time each category is active, so categories that overlap double-count.",
    "overlap: inclusive total minus busy time. idle: no GPU work.",
)
HEADLINE_ROWS = 2


def render_diff(diff: Diff, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(_document(diff), indent=2) + "\n"
    return _markdown(diff) if fmt == "markdown" else _table(diff)


def _label(name: str) -> str:
    return f"{name} (no GPU work)" if name == IDLE else name


def _heading(phase: PhaseDiff) -> str:
    return (
        f"{phase.phase}: {phase.steps_before} vs {phase.steps_after} steps, step time (mean per step) "
        f"{us(phase.step_time_before)} -> {us(phase.step_time_after)} us ({signed(phase.step_time_delta)})"
    )


def _headline(phase: PhaseDiff) -> str:
    changed = [f"{_label(row.name)} {signed(row.delta)} us" for row in phase.rows if row.delta][
        :HEADLINE_ROWS
    ]
    return "largest change: " + (", ".join(changed) or "none")


def _overlap_line(phase: PhaseDiff) -> str:
    overlap = phase.overlap
    return f"overlap {us(overlap.before)} -> {us(overlap.after)} us ({signed(overlap.delta)})"


def _check_line(phase: PhaseDiff) -> str:
    return (
        f"check: rows add up to the step-time change ({signed(phase.step_time_delta)} us), "
        f"residual {us(phase.residual)} us"
    )


def _exclusive_cells(row: Row) -> tuple[str, ...]:
    shares = (percent(row.share_before), percent(row.share_after), signed(row.share_delta_pt))
    return (_label(row.name), us(row.before), us(row.after), signed(row.delta), *shares)


def _inclusive_cells(row: Row) -> tuple[str, ...]:
    return (row.name, us(row.before), us(row.after), signed(row.delta))


def _aligned(header: tuple[str, ...], cells: list[tuple[str, ...]], indent: str) -> list[str]:
    table = [header, *cells]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    return [
        indent
        + row[0].ljust(widths[0])
        + "  "
        + "  ".join(row[i].rjust(widths[i]) for i in range(1, len(row)))
        for row in table
    ]


def _table(diff: Diff) -> str:
    lines = [
        f"before: {diff.before_path}",
        f"after: {diff.after_path}",
        f"device: {diff.device}",
        "",
        *LEGEND,
        "",
    ]
    for phase in diff.phases:
        lines += [_heading(phase), "  " + _headline(phase), "  exclusive"]
        lines += _aligned(EXCLUSIVE_HEADER, [_exclusive_cells(r) for r in phase.rows], "    ")
        lines.append("  inclusive")
        lines += _aligned(INCLUSIVE_HEADER, [_inclusive_cells(r) for r in phase.inclusive_rows], "    ")
        lines += ["  " + _overlap_line(phase), "  " + _check_line(phase), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _markdown_table(header: tuple[str, ...], cells: list[tuple[str, ...]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|---|" + "--:|" * (len(header) - 1)]
    return [*lines, *("| " + " | ".join(row) + " |" for row in cells), ""]


def _markdown(diff: Diff) -> str:
    lines = [f"**before** `{diff.before_path}`, **after** `{diff.after_path}`, **device** {diff.device}", ""]
    lines += [f"_{line}_" for line in LEGEND] + [""]
    for phase in diff.phases:
        lines += [f"## {_heading(phase)}", "", f"**{_headline(phase)}**", "", "### Exclusive", ""]
        lines += _markdown_table(EXCLUSIVE_HEADER, [_exclusive_cells(r) for r in phase.rows])
        lines += ["### Inclusive", ""]
        lines += _markdown_table(INCLUSIVE_HEADER, [_inclusive_cells(r) for r in phase.inclusive_rows])
        lines += [_overlap_line(phase), "", _check_line(phase), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _row_document(row: Row) -> dict:
    document = {
        "category": row.name,
        "before": dec(row.before),
        "after": dec(row.after),
        "delta": dec(row.delta),
    }
    if row.share_delta_pt is not None:
        document |= {
            "share_before": dec(row.share_before),
            "share_after": dec(row.share_after),
            "share_delta_pt": dec(row.share_delta_pt),
        }
    return document


def _phase_document(phase: PhaseDiff) -> dict:
    return {
        "phase": phase.phase,
        "steps_before": phase.steps_before,
        "steps_after": phase.steps_after,
        "step_time_before": dec(phase.step_time_before),
        "step_time_after": dec(phase.step_time_after),
        "step_time_delta": dec(phase.step_time_delta),
        "rows": [_row_document(r) for r in phase.rows],
        "inclusive_rows": [_row_document(r) for r in phase.inclusive_rows],
        "overlap": _row_document(phase.overlap),
        "residual": dec(phase.residual),
    }


def _document(diff: Diff) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": "us",
        "before": str(diff.before_path),
        "after": str(diff.after_path),
        "device": diff.device,
        "options": options_document(diff.options),
        "rules": rules_document(diff.rules),
        "phases": [_phase_document(p) for p in diff.phases],
        "warnings": diff.warnings,
    }
