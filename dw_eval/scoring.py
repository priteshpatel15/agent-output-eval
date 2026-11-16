from __future__ import annotations

from statistics import fmean
from typing import Any

from .models import CORE_DIMENSIONS, DimensionScore, ScoreBlock


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def ratio(value: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return clamp(value / target)


def boolean(value: Any) -> float:
    return 1.0 if value else 0.0


def score_from_ratio(value: float) -> float:
    return round(1 + (4 * clamp(value)), 2)


def build_dimension(
    name: str,
    measures: dict[str, tuple[float, float]],
    evidence: list[str] | None = None,
    checks: dict[str, Any] | None = None,
    confidence: float = 1.0,
    residual: bool = False,
    notes: str = "",
) -> DimensionScore:
    weighted_total = 0.0
    weight_sum = 0.0
    normalized_checks: dict[str, float] = {}
    for key, (value, weight) in measures.items():
        normalized_value = clamp(value)
        normalized_checks[key] = round(normalized_value, 4)
        weighted_total += normalized_value * weight
        weight_sum += weight
    ratio_value = weighted_total / weight_sum if weight_sum else 0.0
    return DimensionScore(
        dimension=name,
        score=score_from_ratio(ratio_value),
        confidence=round(clamp(confidence), 2),
        evidence=evidence or [],
        checks={**(checks or {}), "normalized_measures": normalized_checks},
        residual=residual,
        notes=notes,
    )


def build_score_block(label: str, dimensions: dict[str, DimensionScore]) -> ScoreBlock:
    ordered_scores = [dimensions[name].score for name in CORE_DIMENSIONS if name in dimensions]
    ordered_confidence = [dimensions[name].confidence for name in CORE_DIMENSIONS if name in dimensions]
    average = round(fmean(ordered_scores), 2) if ordered_scores else 0.0
    confidence = round(fmean(ordered_confidence), 2) if ordered_confidence else 0.0
    return ScoreBlock(label=label, dimensions=dimensions, average=average, confidence=confidence)
