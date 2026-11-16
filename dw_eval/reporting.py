from __future__ import annotations

from .models import CORE_DIMENSIONS, EvaluationResult


def _top_evidence(result: EvaluationResult, limit: int = 3) -> list[str]:
    evidence: list[str] = []
    for block in (result.artifact, result.process):
        for dimension in CORE_DIMENSIONS:
            for entry in block.dimensions[dimension].evidence:
                if entry not in evidence:
                    evidence.append(entry)
                if len(evidence) >= limit:
                    return evidence
    return evidence


def _redo_focus(result: EvaluationResult, limit: int = 2) -> list[str]:
    if result.gate_failures:
        return result.gate_failures[:limit]

    candidates: list[tuple[float, str]] = []
    for block in (result.artifact, result.process):
        for dimension in CORE_DIMENSIONS:
            score = block.dimensions[dimension]
            candidates.append(
                (
                    score.score,
                    f"{block.label} {dimension} scored {score.score:.2f}/5",
                )
            )
    candidates.sort(key=lambda item: item[0])
    return [entry for _, entry in candidates[:limit]]


def render_markdown_report(result: EvaluationResult) -> str:
    lines = [
        f"# Evaluation Report: {result.title}",
        "",
        f"- Work type: `{result.work_type}`",
        f"- Generated at: `{result.generated_at}`",
        f"- Overall score: `{result.overall_score}/5`",
        f"- Overall confidence: `{result.overall_confidence}`",
        "",
    ]

    top_evidence = _top_evidence(result)
    redo_focus = _redo_focus(result)

    if result.gate_failures:
        lines.extend(["## Gate Failures", ""])
        for gate in result.gate_failures:
            lines.append(f"- {gate}")
        lines.append("")

    lines.extend(["## Review Summary", ""])
    if top_evidence:
        lines.append("- Top evidence:")
        for entry in top_evidence:
            lines.append(f"  - {entry}")
    else:
        lines.append("- Top evidence: none captured")
    if redo_focus:
        lines.append("- Recommended redo focus:")
        for entry in redo_focus:
            lines.append(f"  - {entry}")
    lines.append("")

    for block in (result.artifact, result.process):
        lines.extend(
            [
                f"## {block.label.title()} Scorecard",
                "",
                f"- Average: `{block.average}/5`",
                f"- Confidence: `{block.confidence}`",
                "",
                "| Dimension | Score | Confidence | Evidence |",
                "|---|---:|---:|---|",
            ]
        )
        for dimension in CORE_DIMENSIONS:
            score = block.dimensions[dimension]
            evidence = "; ".join(score.evidence[:3]) if score.evidence else "-"
            lines.append(
                f"| {dimension} | {score.score:.2f} | {score.confidence:.2f} | {evidence} |"
            )
        lines.append("")

    if result.residual_judgments:
        lines.extend(["## Residual Judgments", ""])
        for note in result.residual_judgments:
            lines.append(f"- {note}")
        lines.append("")

    lines.extend(["## Saved Paths", ""])
    if result.saved_paths:
        for key, value in result.saved_paths.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- Not saved")
    lines.append("")

    return "\n".join(lines)
