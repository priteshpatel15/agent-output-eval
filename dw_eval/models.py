from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CORE_DIMENSIONS = (
    "coverage",
    "correctness",
    "specificity",
    "verification",
    "robustness",
    "closure",
)

WORK_TYPES = (
    "research",
    "design",
    "content",
    "software",
    "outreach",
    "decomposition",
    "procedure",
    "operations",
)


@dataclass
class DimensionScore:
    dimension: str
    score: float
    confidence: float
    evidence: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)
    residual: bool = False
    notes: str = ""


@dataclass
class ScoreBlock:
    label: str
    dimensions: dict[str, DimensionScore]
    average: float
    confidence: float


@dataclass
class EvaluationResult:
    title: str
    work_type: str
    generated_at: str
    artifact_refs: list[str]
    process_trace_refs: list[str]
    artifact: ScoreBlock
    process: ScoreBlock
    overall_score: float
    overall_confidence: float
    gate_failures: list[str] = field(default_factory=list)
    residual_judgments: list[str] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    saved_paths: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
