from __future__ import annotations

import json

from .diff import Diff, PhaseDiff, Row
from .fmt import percent, signed, us
from .render import SCHEMA_VERSION


def render_diff(diff: Diff, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(_diff_document(diff), indent=2) + "\n"
    return _diff_markdown(diff) if fmt == "markdown" else _diff_table(diff)


def _diff_heading(phase: PhaseDiff) -> str:
    return (
        f"{phase.phase}: {phase.steps_before} vs {phase.steps_after} steps, step time "
        f"{us(phase.step_time_before)} -> {us(phase.step_time_after)} us ({signed(phase.step_time_delta)})"
    )


DIFF_HEADER = ("category", "before us", "after us", "delta us", "share before", "share after", "delta pt")


def _diff_cells(row: Row) -> tuple[str, ...]:
    shares = (
        (percent(row.share_before), percent(row.share_after), signed(row.share_delta_pt))
        if row.share_delta_pt is not None
        else ("", "", "")
    )
    return (row.name, us(row.before), us(row.after), signed(row.delta), *shares)


def _overlap_line(phase: PhaseDiff) -> str:
    overlap = phase.overlap
    return f"overlap {us(overlap.before)} -> {us(overlap.after)} us ({signed(overlap.delta)})"


def _residual_line(phase: PhaseDiff) -> str:
    change, residual = signed(phase.step_time_delta), us(phase.residual)
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
