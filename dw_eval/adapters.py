from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable

from .io import utc_now_iso
from .models import CORE_DIMENSIONS, EvaluationResult
from .scoring import boolean, build_dimension, build_score_block, clamp, ratio


def _f(features: dict[str, Any], key: str) -> float:
    value = features.get(key, 0)
    return float(value) if isinstance(value, (int, float, bool)) else 0.0


def _has(features: dict[str, Any], key: str) -> float:
    return boolean(features.get(key))


def _confidence_from_features(features: dict[str, Any], trace: bool = False) -> float:
    if trace:
        base = 0.35
        base += 0.2 if _f(features, "trace_count") else 0.0
        base += 0.2 if _f(features, "typed_event_count") >= 3 else 0.0
        base += 0.15 if _f(features, "save_event_count") >= 1 else 0.0
        base += 0.1 if _f(features, "verification_event_count") >= 1 else 0.0
        return round(clamp(base), 2)
    base = 0.55
    base += 0.15 if _f(features, "word_count") >= 250 else 0.0
    base += 0.15 if _f(features, "heading_count") >= 2 else 0.0
    base += 0.15 if _f(features, "artifact_count") >= 1 else 0.0
    return round(clamp(base), 2)


def _generic_process_dimensions(
    features: dict[str, Any],
    search_target: float = 1,
    source_target: float = 1,
    option_target: float = 1,
    verification_target: float = 1,
    approval_target: float = 0,
    save_target: float = 1,
    revision_target: float = 1,
) -> dict[str, Any]:
    confidence = _confidence_from_features(features, trace=True)
    blocker_count = _f(features, "blocker_event_count")
    resume_count = _f(features, "resume_event_count")
    error_free = 1.0 if _f(features, "error_event_count") == 0 else 0.0
    blocker_recovery = 1.0 if blocker_count == 0 or resume_count >= blocker_count else 0.0

    dimensions = {
        "coverage": build_dimension(
            "coverage",
            {
                "event_count": (ratio(_f(features, "event_count"), 8), 0.5),
                "save_event_count": (ratio(_f(features, "save_event_count"), save_target), 0.5),
            },
            evidence=[
                f"Trace count: {int(_f(features, 'trace_count'))}",
                f"Event count: {int(_f(features, 'event_count'))}",
            ],
            confidence=confidence,
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "error_free": (error_free, 0.6),
                "success_signals": (ratio(_f(features, "success_count"), 2), 0.4),
            },
            evidence=[f"Error events: {int(_f(features, 'error_event_count'))}"],
            confidence=confidence,
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "typed_events": (ratio(_f(features, "typed_event_count"), 4), 0.6),
                "save_events": (ratio(_f(features, "save_event_count"), save_target), 0.4),
            },
            evidence=[f"Typed events: {int(_f(features, 'typed_event_count'))}"],
            confidence=confidence,
        ),
        "verification": build_dimension(
            "verification",
            {
                "verification_events": (ratio(_f(features, "verification_event_count"), verification_target), 0.6),
                "approval_events": (
                    ratio(_f(features, "approval_event_count") + _f(features, "approved_count"), max(approval_target, 1)),
                    0.4,
                ),
            },
            evidence=[
                f"Verification events: {int(_f(features, 'verification_event_count'))}",
                f"Approval events: {int(_f(features, 'approval_event_count'))}",
            ],
            confidence=confidence,
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "revision_events": (ratio(_f(features, "revision_event_count"), revision_target), 0.5),
                "blocker_recovery": (blocker_recovery, 0.5),
            },
            evidence=[
                f"Revision events: {int(_f(features, 'revision_event_count'))}",
                f"Blockers: {int(blocker_count)} / resumes: {int(resume_count)}",
            ],
            confidence=confidence,
        ),
        "closure": build_dimension(
            "closure",
            {
                "save_event_count": (ratio(_f(features, "save_event_count"), save_target), 0.6),
                "verification_marker": (ratio(_f(features, "trace_receipt_mentions"), 1), 0.4),
            },
            evidence=[f"Save events: {int(_f(features, 'save_event_count'))}"],
            confidence=confidence,
        ),
    }

    if search_target:
        dimensions["coverage"].checks["search_event_count"] = _f(features, "search_event_count")
        dimensions["coverage"].checks["source_event_count"] = _f(features, "source_event_count")
        dimensions["coverage"].checks["option_event_count"] = _f(features, "option_event_count")
    return dimensions


@dataclass
class AdapterOutcome:
    artifact_dimensions: dict[str, Any]
    process_dimensions: dict[str, Any]
    gate_failures: list[str]
    residual_judgments: list[str]


def _research_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = _confidence_from_features(features)
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "executive_summary": (_has(features, "has_executive_summary_section"), 0.2),
                "findings": (_has(features, "has_findings_section"), 0.2),
                "analysis": (_has(features, "has_analysis_section"), 0.2),
                "recommendations": (_has(features, "has_recommendations_section"), 0.2),
                "sources": (_has(features, "has_sources_section"), 0.2),
            },
            evidence=[f"Source count: {int(_f(features, 'url_count'))}"],
            confidence=confidence,
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "citations": (ratio(_f(features, "citation_like_count"), 7), 0.4),
                "primary_sources": (ratio(_f(features, "primary_source_count"), 2), 0.3),
                "domain_diversity": (ratio(_f(features, "unique_domain_count"), 4), 0.3),
            },
            evidence=[f"Primary sources: {int(_f(features, 'primary_source_count'))}"],
            confidence=confidence,
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "recommendations": (ratio(_f(features, "recommendation_count"), 3), 0.4),
                "owners": (ratio(_f(features, "owner_marker_count"), 1), 0.3),
                "timeline": (ratio(_f(features, "timeline_marker_count"), 1), 0.3),
            },
            evidence=[f"Recommendation markers: {int(_f(features, 'recommendation_count'))}"],
            confidence=confidence,
        ),
        "verification": build_dimension(
            "verification",
            {
                "source_list": (_has(features, "has_sources_section"), 0.4),
                "citations": (ratio(_f(features, "citation_like_count"), 7), 0.3),
                "recency_markers": (ratio(_f(features, "date_count"), 3), 0.3),
            },
            evidence=[f"Date mentions: {int(_f(features, 'date_count'))}"],
            confidence=confidence,
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "analysis": (_has(features, "has_analysis_section"), 0.3),
                "risk_markers": (ratio(_f(features, "risk_mention_count"), 1), 0.3),
                "option_markers": (ratio(_f(features, "named_option_count"), 1), 0.4),
            },
            evidence=[f"Risk mentions: {int(_f(features, 'risk_mention_count'))}"],
            confidence=confidence,
        ),
        "closure": build_dimension(
            "closure",
            {
                "recommendations": (_has(features, "has_recommendations_section"), 0.5),
                "closure_markers": (ratio(_f(features, "next_step_count"), 1), 0.5),
            },
            evidence=[f"Next-step markers: {int(_f(features, 'next_step_count'))}"],
            confidence=confidence,
        ),
    }
    gates = []
    if _f(features, "citation_like_count") < 3:
        gates.append("Research output lacks enough citations for a reliable evidence base.")
    process = _generic_process_dimensions(trace, search_target=3, source_target=5, verification_target=2, save_target=1, revision_target=1)
    return AdapterOutcome(artifact, process, gates, [])


def _design_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = _confidence_from_features(features)
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "requirements": (_has(features, "has_requirements_section"), 0.25),
                "constraints": (_has(features, "has_constraints_section"), 0.2),
                "options": (_has(features, "has_options_section"), 0.15),
                "implementation": (_has(features, "has_implementation_section"), 0.2),
                "risks": (_has(features, "has_risks_section"), 0.2),
            },
            confidence=confidence,
            evidence=[f"Heading count: {int(_f(features, 'heading_count'))}"],
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "requirements": (_has(features, "has_requirements_section"), 0.35),
                "constraints": (_has(features, "has_constraints_section"), 0.35),
                "decision_matrix": (_has(features, "has_decision_matrix_section"), 0.3),
            },
            confidence=confidence,
            evidence=[f"Constraint mentions: {int(_f(features, 'constraint_mention_count'))}"],
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "dependencies": (ratio(_f(features, "dependency_mention_count"), 2), 0.3),
                "timeline": (ratio(_f(features, "timeline_marker_count"), 2), 0.3),
                "owners": (ratio(_f(features, "owner_marker_count"), 1), 0.2),
                "implementation": (_has(features, "has_implementation_section"), 0.2),
            },
            confidence=confidence,
            evidence=[f"Dependency mentions: {int(_f(features, 'dependency_mention_count'))}"],
        ),
        "verification": build_dimension(
            "verification",
            {
                "decision_matrix": (_has(features, "has_decision_matrix_section"), 0.45),
                "requirements": (_has(features, "has_requirements_section"), 0.25),
                "risks": (_has(features, "has_risks_section"), 0.3),
            },
            confidence=confidence,
            evidence=[f"Option markers: {int(_f(features, 'named_option_count'))}"],
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "risks": (_has(features, "has_risks_section"), 0.4),
                "option_markers": (ratio(_f(features, "named_option_count"), 2), 0.3),
                "edge_case_markers": (ratio(_f(features, "risk_mention_count"), 2), 0.3),
            },
            confidence=confidence,
            evidence=[f"Risk mentions: {int(_f(features, 'risk_mention_count'))}"],
        ),
        "closure": build_dimension(
            "closure",
            {
                "implementation": (_has(features, "has_implementation_section"), 0.4),
                "closure_markers": (ratio(_f(features, "next_step_count"), 1), 0.3),
                "owner_markers": (ratio(_f(features, "owner_marker_count"), 1), 0.3),
            },
            confidence=confidence,
            evidence=[f"Timeline markers: {int(_f(features, 'timeline_marker_count'))}"],
        ),
    }
    gates = []
    if not _has(features, "has_requirements_section"):
        gates.append("Design artifact is missing a requirements section.")
    if not _has(features, "has_implementation_section"):
        gates.append("Design artifact is missing an implementation path.")
    process = _generic_process_dimensions(trace, option_target=2, verification_target=1, save_target=1, revision_target=1)
    return AdapterOutcome(artifact, process, gates, [])


def _content_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = _confidence_from_features(features)
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "summary": (_has(features, "has_executive_summary_section"), 0.2),
                "recommendations_or_cta": (ratio(_f(features, "ask_count") + _f(features, "recommendation_count"), 1), 0.4),
                "required_sections": (features.get("required_section_coverage", 0.5), 0.4),
            },
            confidence=confidence,
            evidence=[f"Word count: {int(_f(features, 'word_count'))}"],
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "must_include_terms": (
                    clamp(features.get("must_include_hits", 0) / max(len(context.get("must_include_terms", [])), 1))
                    if context.get("must_include_terms")
                    else 0.5,
                    0.5,
                ),
                "citation_support": (0.5 if not context.get("requires_citations") else ratio(_f(features, "citation_like_count"), 3), 0.5),
            },
            confidence=confidence * 0.9,
            evidence=[f"Must-include hits: {int(_f(features, 'must_include_hits'))}"],
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "clear_ask": (ratio(_f(features, "ask_count"), 1), 0.4),
                "timeline": (ratio(_f(features, "timeline_marker_count"), 1), 0.3),
                "audience_terms": (
                    clamp(_f(features, "audience_term_hits") / max(len(context.get("audience_terms", [])), 1))
                    if context.get("audience_terms")
                    else 0.5,
                    0.3,
                ),
            },
            confidence=confidence,
            evidence=[f"Ask count: {int(_f(features, 'ask_count'))}"],
        ),
        "verification": build_dimension(
            "verification",
            {
                "claim_support": (0.5 if _f(features, "citation_like_count") == 0 else ratio(_f(features, "citation_like_count"), 2), 0.5),
                "specific_dates": (ratio(_f(features, "date_count"), 1), 0.25),
                "specific_links": (ratio(_f(features, "url_count"), 1), 0.25),
            },
            confidence=confidence * 0.85,
            evidence=[f"Citation-like count: {int(_f(features, 'citation_like_count'))}"],
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "questions": (ratio(_f(features, "question_count"), 1), 0.3),
                "follow_up": (_has(features, "has_follow_up_section"), 0.35),
                "risk_markers": (ratio(_f(features, "risk_mention_count"), 1), 0.35),
            },
            confidence=confidence * 0.9,
            evidence=[f"Question count: {int(_f(features, 'question_count'))}"],
        ),
        "closure": build_dimension(
            "closure",
            {
                "cta": (ratio(_f(features, "ask_count"), 1), 0.6),
                "next_steps": (ratio(_f(features, "next_step_count"), 1), 0.4),
            },
            confidence=confidence,
            evidence=[f"Next-step markers: {int(_f(features, 'next_step_count'))}"],
        ),
    }
    process = _generic_process_dimensions(trace, verification_target=1, save_target=1, revision_target=1)
    residuals = [
        "Originality should be reviewed through pairwise human preference, not treated as deterministic.",
        "Emotional resonance should be treated as residual judgment or delayed outcome, not a script-only metric.",
    ]
    return AdapterOutcome(artifact, process, [], residuals)


def _software_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = _confidence_from_features(features)
    test_pass = 1.0 if _f(trace, "test_event_count") and _f(trace, "error_event_count") == 0 else 0.5 if _f(features, "artifact_file_count") else 0.0
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "files_present": (ratio(_f(features, "artifact_file_count"), 2), 0.3),
                "tests_or_docs": (
                    clamp((_f(features, "artifact_file_count") > 0 and ("test" in str(context).lower() or _f(trace, "test_event_count") > 0)) * 1.0),
                    0.35,
                ),
                "implementation_details": (_has(features, "has_implementation_section"), 0.35),
            },
            confidence=confidence,
            evidence=[f"Artifact file count: {int(_f(features, 'artifact_file_count'))}"],
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "tests": (test_pass, 0.5),
                "build": (1.0 if _f(trace, "build_event_count") and _f(trace, "error_event_count") == 0 else 0.5 if _f(trace, "build_event_count") else 0.3, 0.25),
                "lint": (1.0 if _f(trace, "lint_event_count") and _f(trace, "error_event_count") == 0 else 0.5 if _f(trace, "lint_event_count") else 0.3, 0.25),
            },
            confidence=max(confidence, _confidence_from_features(trace, trace=True)),
            evidence=[
                f"Test events: {int(_f(trace, 'test_event_count'))}",
                f"Build events: {int(_f(trace, 'build_event_count'))}",
            ],
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "implementation": (_has(features, "has_implementation_section"), 0.4),
                "verification_markers": (ratio(_f(features, "verification_marker_count"), 1), 0.3),
                "closure_markers": (ratio(_f(features, "next_step_count"), 1), 0.3),
            },
            confidence=confidence,
            evidence=[f"Verification markers: {int(_f(features, 'verification_marker_count'))}"],
        ),
        "verification": build_dimension(
            "verification",
            {
                "tests": (test_pass, 0.45),
                "verification_events": (ratio(_f(trace, "verification_event_count"), 1), 0.35),
                "coverage": (ratio(_f(trace, "reported_coverage_avg"), 70), 0.2),
            },
            confidence=max(confidence, _confidence_from_features(trace, trace=True)),
            evidence=[f"Reported coverage average: {round(_f(trace, 'reported_coverage_avg'), 2)}"],
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "revision": (ratio(_f(trace, "revision_event_count"), 1), 0.3),
                "error_free": (1.0 if _f(trace, "error_event_count") == 0 else 0.0, 0.4),
                "dependency_mentions": (ratio(_f(features, "dependency_mention_count"), 1), 0.3),
            },
            confidence=max(confidence, _confidence_from_features(trace, trace=True)),
            evidence=[f"Error events: {int(_f(trace, 'error_event_count'))}"],
        ),
        "closure": build_dimension(
            "closure",
            {
                "save_events": (ratio(_f(trace, "save_event_count"), 1), 0.4),
                "next_steps": (ratio(_f(features, "next_step_count"), 1), 0.3),
                "verification": (ratio(_f(trace, "verification_event_count"), 1), 0.3),
            },
            confidence=max(confidence, _confidence_from_features(trace, trace=True)),
            evidence=[f"Save events: {int(_f(trace, 'save_event_count'))}"],
        ),
    }
    gates = []
    if _f(trace, "error_event_count") > 0:
        gates.append("Software trace contains error events; do not mark the work complete.")
    if _f(trace, "test_event_count") == 0:
        gates.append("Software evaluation is missing explicit test evidence.")
    process = _generic_process_dimensions(trace, verification_target=1, save_target=1, revision_target=1)
    return AdapterOutcome(artifact, process, gates, [])


def _outreach_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = _confidence_from_features(features)
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "message": (ratio(_f(features, "word_count"), 80), 0.4),
                "agenda_or_followup": (_has(features, "has_follow_up_section") + _has(features, "has_agenda_section"), 0.6),
            },
            confidence=confidence,
            evidence=[f"Word count: {int(_f(features, 'word_count'))}"],
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "must_include_terms": (
                    clamp(_f(features, "must_include_hits") / max(len(context.get("must_include_terms", [])), 1))
                    if context.get("must_include_terms")
                    else 0.5,
                    0.5,
                ),
                "audience_terms": (
                    clamp(_f(features, "audience_term_hits") / max(len(context.get("audience_terms", [])), 1))
                    if context.get("audience_terms")
                    else 0.5,
                    0.5,
                ),
            },
            confidence=confidence * 0.85,
            evidence=[f"Audience term hits: {int(_f(features, 'audience_term_hits'))}"],
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "clear_ask": (ratio(_f(features, "ask_count"), 1), 0.4),
                "meeting_windows": (ratio(_f(features, "meeting_window_count"), 2), 0.3),
                "follow_up": (_has(features, "has_follow_up_section"), 0.3),
            },
            confidence=confidence,
            evidence=[f"Meeting windows: {int(_f(features, 'meeting_window_count'))}"],
        ),
        "verification": build_dimension(
            "verification",
            {
                "follow_up": (_has(features, "has_follow_up_section"), 0.5),
                "timeline": (ratio(_f(features, "timeline_marker_count"), 1), 0.5),
            },
            confidence=confidence,
            evidence=[f"Follow-up section: {bool(features.get('has_follow_up_section'))}"],
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "follow_up": (_has(features, "has_follow_up_section"), 0.4),
                "agenda": (_has(features, "has_agenda_section"), 0.3),
                "questions": (ratio(_f(features, "question_count"), 1), 0.3),
            },
            confidence=confidence,
            evidence=[f"Agenda section: {bool(features.get('has_agenda_section'))}"],
        ),
        "closure": build_dimension(
            "closure",
            {
                "clear_ask": (ratio(_f(features, "ask_count"), 1), 0.5),
                "next_step": (ratio(_f(features, "next_step_count"), 1), 0.5),
            },
            confidence=confidence,
            evidence=[f"Ask count: {int(_f(features, 'ask_count'))}"],
        ),
    }
    gates = []
    if _f(features, "ask_count") < 1:
        gates.append("Outreach package does not contain a clear ask.")
    process = _generic_process_dimensions(trace, approval_target=1, verification_target=1, save_target=1, revision_target=1)
    residuals = ["Relationship nuance and warmth should be reviewed by preference or outcome, not reduced to a script-only score."]
    return AdapterOutcome(artifact, process, gates, residuals)


def _decomposition_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = _confidence_from_features(features)
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "bullets": (ratio(_f(features, "bullet_count"), 5), 0.3),
                "dependencies": (_has(features, "has_dependencies_section"), 0.2),
                "acceptance": (_has(features, "has_acceptance_criteria_section"), 0.2),
                "implementation": (_has(features, "has_implementation_section"), 0.15),
                "closure": (_has(features, "has_closure_section"), 0.15),
            },
            confidence=confidence,
            evidence=[f"Bullet count: {int(_f(features, 'bullet_count'))}"],
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "dependencies": (ratio(_f(features, "dependency_mention_count"), 2), 0.4),
                "acceptance": (ratio(_f(features, "acceptance_criteria_mention_count"), 2), 0.3),
                "owners": (ratio(_f(features, "owner_marker_count"), 1), 0.3),
            },
            confidence=confidence * 0.9,
            evidence=[f"Acceptance mentions: {int(_f(features, 'acceptance_criteria_mention_count'))}"],
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "owners": (ratio(_f(features, "owner_marker_count"), 1), 0.25),
                "timeline": (ratio(_f(features, "timeline_marker_count"), 1), 0.25),
                "dependencies": (ratio(_f(features, "dependency_mention_count"), 2), 0.25),
                "acceptance": (ratio(_f(features, "acceptance_criteria_mention_count"), 2), 0.25),
            },
            confidence=confidence,
            evidence=[f"Owner markers: {int(_f(features, 'owner_marker_count'))}"],
        ),
        "verification": build_dimension(
            "verification",
            {
                "acceptance": (_has(features, "has_acceptance_criteria_section"), 0.5),
                "closure_markers": (ratio(_f(features, "next_step_count"), 1), 0.5),
            },
            confidence=confidence,
            evidence=[f"Closure markers: {int(_f(features, 'next_step_count'))}"],
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "risk_markers": (ratio(_f(features, "risk_mention_count"), 1), 0.4),
                "dependencies": (_has(features, "has_dependencies_section"), 0.3),
                "coverage": (ratio(_f(features, "bullet_count"), 8), 0.3),
            },
            confidence=confidence,
            evidence=[f"Dependency section: {bool(features.get('has_dependencies_section'))}"],
        ),
        "closure": build_dimension(
            "closure",
            {
                "acceptance": (_has(features, "has_acceptance_criteria_section"), 0.5),
                "next_steps": (ratio(_f(features, "next_step_count"), 1), 0.5),
            },
            confidence=confidence,
            evidence=[f"Acceptance section: {bool(features.get('has_acceptance_criteria_section'))}"],
        ),
    }
    gates = []
    if not _has(features, "has_acceptance_criteria_section"):
        gates.append("Decomposition is missing acceptance criteria / done markers.")
    process = _generic_process_dimensions(trace, option_target=1, verification_target=1, save_target=1, revision_target=1)
    return AdapterOutcome(artifact, process, gates, [])


def _procedure_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = _confidence_from_features(features)
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "prerequisites": (_has(features, "has_prerequisites_section"), 0.2),
                "contingencies": (_has(features, "has_contingencies_section"), 0.2),
                "verification": (_has(features, "has_verification_section"), 0.2),
                "closure": (_has(features, "has_closure_section"), 0.2),
                "steps": (ratio(_f(features, "bullet_count"), 4), 0.2),
            },
            confidence=confidence,
            evidence=[f"Step-like bullets: {int(_f(features, 'bullet_count'))}"],
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "resources": (ratio(_f(features, "url_count") + _f(features, "phone_count"), 2), 0.5),
                "verification": (_has(features, "has_verification_section"), 0.5),
            },
            confidence=confidence,
            evidence=[f"URLs + phones: {int(_f(features, 'url_count') + _f(features, 'phone_count'))}"],
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "resources": (ratio(_f(features, "url_count") + _f(features, "phone_count"), 2), 0.35),
                "time_markers": (ratio(_f(features, "time_count"), 1), 0.25),
                "date_markers": (ratio(_f(features, "date_count"), 1), 0.2),
                "bullets": (ratio(_f(features, "bullet_count"), 4), 0.2),
            },
            confidence=confidence,
            evidence=[f"Time markers: {int(_f(features, 'time_count'))}"],
        ),
        "verification": build_dimension(
            "verification",
            {
                "verification": (_has(features, "has_verification_section"), 0.5),
                "confirmation_markers": (ratio(_f(features, "confirmation_id_count"), 1), 0.5),
            },
            confidence=confidence,
            evidence=[f"Confirmation markers: {int(_f(features, 'confirmation_id_count'))}"],
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "contingencies": (_has(features, "has_contingencies_section"), 0.45),
                "contingency_markers": (ratio(_f(features, "contingency_count"), 2), 0.3),
                "prerequisites": (_has(features, "has_prerequisites_section"), 0.25),
            },
            confidence=confidence,
            evidence=[f"Contingency markers: {int(_f(features, 'contingency_count'))}"],
        ),
        "closure": build_dimension(
            "closure",
            {
                "next_steps": (ratio(_f(features, "next_step_count"), 1), 0.5),
                "verification": (_has(features, "has_verification_section"), 0.5),
            },
            confidence=confidence,
            evidence=[f"Next-step markers: {int(_f(features, 'next_step_count'))}"],
        ),
    }
    gates = []
    if not _has(features, "has_prerequisites_section"):
        gates.append("Procedure is missing prerequisites.")
    if not _has(features, "has_contingencies_section"):
        gates.append("Procedure is missing contingency handling.")
    process = _generic_process_dimensions(trace, verification_target=1, save_target=1, revision_target=1)
    return AdapterOutcome(artifact, process, gates, [])


def _operations_adapter(features: dict[str, Any], trace: dict[str, Any], context: dict[str, Any]) -> AdapterOutcome:
    confidence = max(_confidence_from_features(features), _confidence_from_features(trace, trace=True))
    artifact = {
        "coverage": build_dimension(
            "coverage",
            {
                "verification": (_has(features, "has_verification_section"), 0.25),
                "closure": (_has(features, "has_closure_section"), 0.25),
                "confirmation_markers": (ratio(_f(features, "confirmation_id_count"), 1), 0.25),
                "next_steps": (ratio(_f(features, "next_step_count"), 1), 0.25),
            },
            confidence=confidence,
            evidence=[f"Confirmation markers: {int(_f(features, 'confirmation_id_count'))}"],
        ),
        "correctness": build_dimension(
            "correctness",
            {
                "zone_declared": (1.0 if context.get("zone") else 0.5, 0.3),
                "verification_markers": (ratio(_f(features, "verification_marker_count"), 1), 0.4),
                "approval_trace": (ratio(_f(trace, "approval_event_count") + _f(trace, "approved_count"), 1), 0.3),
            },
            confidence=confidence,
            evidence=[f"Approval events: {int(_f(trace, 'approval_event_count'))}"],
        ),
        "specificity": build_dimension(
            "specificity",
            {
                "urls": (ratio(_f(features, "url_count"), 1), 0.25),
                "confirmation": (ratio(_f(features, "confirmation_id_count"), 1), 0.35),
                "timeline": (ratio(_f(features, "timeline_marker_count"), 1), 0.2),
                "closure": (ratio(_f(features, "next_step_count"), 1), 0.2),
            },
            confidence=confidence,
            evidence=[f"URLs: {int(_f(features, 'url_count'))}"],
        ),
        "verification": build_dimension(
            "verification",
            {
                "verification_events": (ratio(_f(trace, "verification_event_count"), 1), 0.35),
                "verification_markers": (ratio(_f(features, "verification_marker_count"), 1), 0.35),
                "confirmation_markers": (ratio(_f(features, "confirmation_id_count"), 1), 0.3),
            },
            confidence=confidence,
            evidence=[f"Verification events: {int(_f(trace, 'verification_event_count'))}"],
        ),
        "robustness": build_dimension(
            "robustness",
            {
                "blocker_recovery": (1.0 if _f(trace, "blocker_event_count") == 0 or _f(trace, "resume_event_count") >= _f(trace, "blocker_event_count") else 0.0, 0.45),
                "revision_events": (ratio(_f(trace, "revision_event_count"), 1), 0.25),
                "contingency_markers": (ratio(_f(features, "contingency_count"), 1), 0.3),
            },
            confidence=confidence,
            evidence=[f"Blockers: {int(_f(trace, 'blocker_event_count'))}"],
        ),
        "closure": build_dimension(
            "closure",
            {
                "save_events": (ratio(_f(trace, "save_event_count"), 1), 0.25),
                "receipts": (ratio(_f(trace, "trace_receipt_mentions") + _f(features, "confirmation_id_count"), 1), 0.5),
                "next_steps": (ratio(_f(features, "next_step_count"), 1), 0.25),
            },
            confidence=confidence,
            evidence=[f"Receipt mentions: {int(_f(trace, 'trace_receipt_mentions'))}"],
        ),
    }
    gates = []
    if _f(trace, "approval_event_count") + _f(trace, "approved_count") < 1 and context.get("action_class") in {"send", "submit", "book", "delete", "pay"}:
        gates.append("Operational trace is missing an approval checkpoint for a risky action.")
    if _f(features, "verification_marker_count") < 1 and _f(trace, "verification_event_count") < 1:
        gates.append("Operational work lacks post-action verification evidence.")
    process = _generic_process_dimensions(trace, verification_target=1, approval_target=1, save_target=1, revision_target=1)
    return AdapterOutcome(artifact, process, gates, [])


ADAPTERS: dict[str, Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], AdapterOutcome]] = {
    "research": _research_adapter,
    "design": _design_adapter,
    "content": _content_adapter,
    "software": _software_adapter,
    "outreach": _outreach_adapter,
    "decomposition": _decomposition_adapter,
    "procedure": _procedure_adapter,
    "operations": _operations_adapter,
}


def _personal_first_gate_failures(
    artifact_features: dict[str, Any],
    trace_features: dict[str, Any],
    context: dict[str, Any],
) -> list[str]:
    gates: list[str] = []
    execution_mode = context.get("execution_mode")
    action_class = context.get("action_class")
    approval_required = bool(context.get("approval_required"))
    requires_clarification = bool(context.get("requires_clarification"))

    has_next_human_action = bool(
        artifact_features.get("has_next_human_action_section")
        or artifact_features.get("next_step_count")
    )
    has_clarifying_questions = bool(
        artifact_features.get("has_clarifying_questions_section")
        or artifact_features.get("clarifying_question_count")
    )

    if execution_mode and not (has_next_human_action or has_clarifying_questions):
        gates.append("Output must state the next human action or ask clarifying questions for the execution mode.")
    if requires_clarification and not has_clarifying_questions:
        gates.append("Blocking ambiguity was declared but the output does not ask clarifying questions.")
    if execution_mode in {"Review/Approve", "Decide"} and not (
        artifact_features.get("has_decision_needed_section") or has_next_human_action or has_clarifying_questions
    ):
        gates.append("Decision-mode work must state the decision needed or the next human action.")
    risky_action = action_class in {
        "send",
        "submit",
        "book",
        "delete",
        "archive",
        "move",
        "pay",
        "financial",
        "medical",
        "high-risk",
    }
    approval_events = _f(trace_features, "approval_event_count") + _f(trace_features, "approved_count")
    if (approval_required or risky_action) and approval_events < 1 and not has_clarifying_questions:
        gates.append("Approval-gated work must record an approval checkpoint or ask before the risky step.")
    return gates


def evaluate_work_type(
    work_type: str,
    title: str,
    artifact_refs: list[str],
    process_trace_refs: list[str],
    artifact_features: dict[str, Any],
    trace_features: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> EvaluationResult:
    context = context or {}
    if work_type not in ADAPTERS:
        raise SystemExit(f"Unsupported work type: {work_type}")

    outcome = ADAPTERS[work_type](artifact_features, trace_features, context)
    artifact_block = build_score_block("artifact", outcome.artifact_dimensions)
    process_block = build_score_block("process", outcome.process_dimensions)
    overall_score = round(fmean([artifact_block.average, process_block.average]), 2)
    overall_confidence = round(fmean([artifact_block.confidence, process_block.confidence]), 2)

    metadata = {
        "adapter_version": "2026-03-08",
        "context_keys": sorted(context.keys()),
    }

    gate_failures = [
        *outcome.gate_failures,
        *_personal_first_gate_failures(artifact_features, trace_features, context),
    ]

    return EvaluationResult(
        title=title,
        work_type=work_type,
        generated_at=utc_now_iso(),
        artifact_refs=artifact_refs,
        process_trace_refs=process_trace_refs,
        artifact=artifact_block,
        process=process_block,
        overall_score=overall_score,
        overall_confidence=overall_confidence,
        gate_failures=gate_failures,
        residual_judgments=outcome.residual_judgments,
        features={"artifact": artifact_features, "process": trace_features},
        metadata=metadata,
    )
