from __future__ import annotations

import re
from collections import Counter
from statistics import fmean
from typing import Any
from urllib.parse import urlparse

from .io import ArtifactInput, TraceInput


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
URL_RE = re.compile(r"https?://[^\s)>\]]+")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+\d{1,2}\s*)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm|AM|PM)\b")

SECTION_PATTERNS = {
    "executive_summary": (r"executive summary", r"\bsummary\b"),
    "findings": (r"key findings", r"\bfindings\b"),
    "analysis": (r"\banalysis\b", r"\bdiscussion\b"),
    "recommendations": (r"\brecommendations?\b", r"\bnext steps\b"),
    "sources": (r"\bsources?\b", r"\breferences\b"),
    "requirements": (r"\brequirements?\b",),
    "constraints": (r"\bconstraints?\b",),
    "options": (r"\boptions?\b", r"\balternatives?\b"),
    "decision_matrix": (r"decision matrix", r"trade[- ]?off"),
    "implementation": (r"implementation", r"\broadmap\b", r"\bphases?\b"),
    "risks": (r"\brisks?\b", r"\bmitigations?\b", r"\bedge cases?\b"),
    "agenda": (r"\bagenda\b",),
    "talking_points": (r"talking points",),
    "follow_up": (r"follow[- ]?up",),
    "acceptance_criteria": (r"acceptance criteria", r"definition of done", r"\bdone criteria\b"),
    "dependencies": (r"\bdependencies?\b", r"\bcritical path\b"),
    "prerequisites": (r"\bprerequisites?\b",),
    "contingencies": (r"\bcontingenc", r"\bif .* happens\b", r"\bfallback\b", r"\bbackup plan\b"),
    "verification": (r"\bverification\b", r"\bverified\b", r"\breceipt\b", r"\bconfirmation\b"),
    "closure": (r"\bnext exact step\b", r"\bremaining follow-up\b", r"\bready for\b"),
    "board_category": (r"\bboard category\b", r"\bcategory\b"),
    "execution_mode": (r"\bexecution mode\b", r"\bmode\b"),
    "next_human_action": (r"\bnext human action\b", r"\bwaiting on you\b"),
    "decision_needed": (r"\bdecision needed\b", r"\bdecision required\b"),
    "clarifying_questions": (r"\bclarifying questions?\b", r"\bquestions? for\b"),
}

PRIMARY_DOMAIN_HINTS = (
    ".gov",
    ".edu",
    "openai.com",
    "docs.",
    "developer.",
    "github.com",
    "wikipedia.org",
    "arxiv.org",
    "aclanthology.org",
)


def _extract_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "_preamble"
    sections[current] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            current = match.group(2).strip().lower()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def _count_list_items(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"^\s*(?:[-*+]|\d+\.)\s+", line))


def _count_sentences_with(pattern: str, text: str) -> int:
    regex = re.compile(pattern, re.IGNORECASE)
    return sum(1 for sentence in re.split(r"(?<=[.!?])\s+", text) if regex.search(sentence))


def _section_matches(sections: dict[str, str], pattern_group: tuple[str, ...]) -> list[str]:
    hits = []
    for heading, body in sections.items():
        source = f"{heading}\n{body}".lower()
        if any(re.search(pattern, source, re.IGNORECASE) for pattern in pattern_group):
            hits.append(heading)
    return hits


def _domain_counts(urls: list[str]) -> tuple[list[str], int]:
    domains: list[str] = []
    primary_count = 0
    for url in urls:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if not domain:
            continue
        domains.append(domain)
        if any(hint in domain for hint in PRIMARY_DOMAIN_HINTS):
            primary_count += 1
    return domains, primary_count


def extract_artifact_features(artifacts: list[ArtifactInput], context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    combined_text = "\n\n".join(artifact.text for artifact in artifacts)
    sections = _extract_markdown_sections(combined_text)
    urls = URL_RE.findall(combined_text)
    domains, primary_source_count = _domain_counts(urls)
    normalized_text = combined_text.lower()
    audience_terms = [term.lower() for term in context.get("audience_terms", [])]
    required_sections = [section.lower() for section in context.get("required_sections", [])]
    must_include_terms = [term.lower() for term in context.get("must_include_terms", [])]

    feature_map: dict[str, Any] = {
        "artifact_count": len(artifacts),
        "artifact_file_count": sum(artifact.file_count for artifact in artifacts),
        "text_length": len(combined_text),
        "word_count": len(combined_text.split()),
        "heading_count": len(sections) - 1 if "_preamble" in sections else len(sections),
        "bullet_count": _count_list_items(combined_text),
        "table_count": combined_text.count("|"),
        "code_block_count": combined_text.count("```") // 2,
        "url_count": len(urls),
        "unique_domain_count": len(set(domains)),
        "primary_source_count": primary_source_count,
        "email_count": len(EMAIL_RE.findall(combined_text)),
        "phone_count": len(PHONE_RE.findall(combined_text)),
        "date_count": len(DATE_RE.findall(combined_text)),
        "time_count": len(TIME_RE.findall(combined_text)),
        "question_count": combined_text.count("?"),
        "clarifying_question_count": _count_sentences_with(r"\?", combined_text)
        if _section_matches(sections, SECTION_PATTERNS["clarifying_questions"])
        else 0,
        "citation_like_count": len(urls) + len(re.findall(r"\[[^\]]+\]\([^)]+\)", combined_text)),
        "checklist_item_count": sum(1 for line in combined_text.splitlines() if re.match(r"^\s*[-*]\s+\[[ xX]\]", line)),
        "named_option_count": _count_sentences_with(r"\boption\b|\balternative\b", combined_text),
        "risk_mention_count": _count_sentences_with(r"\brisk\b|\bmitigat", combined_text),
        "constraint_mention_count": _count_sentences_with(r"\bconstraint\b|\blimit", combined_text),
        "dependency_mention_count": _count_sentences_with(r"\bdependenc", combined_text),
        "acceptance_criteria_mention_count": _count_sentences_with(r"acceptance criteria|definition of done|done criteria", combined_text),
        "recommendation_count": _count_sentences_with(r"\brecommend\b|\bnext step\b|\byou should\b", combined_text),
        "owner_marker_count": _count_sentences_with(r"\bowner\b|\bassign\b|\bresponsible\b", combined_text),
        "timeline_marker_count": _count_sentences_with(r"\bby\b\s+\w+|\bdue\b|\bdeadline\b|\bETA\b", combined_text),
        "impact_marker_count": _count_sentences_with(r"\bimpact\b|\bbenefit\b|\bvalue\b", combined_text),
        "effort_marker_count": _count_sentences_with(r"\beffort\b|\bcomplexity\b|\bestimate\b", combined_text),
        "ask_count": _count_sentences_with(r"\bplease\b|\bcan you\b|\bwould you\b|\blet me know\b", combined_text),
        "meeting_window_count": _count_sentences_with(r"\bmonday\b|\btuesday\b|\bwednesday\b|\bthursday\b|\bfriday\b|\bnext week\b", combined_text),
        "verification_marker_count": _count_sentences_with(r"\bverified\b|\bconfirmation\b|\breceipt\b|\bproof\b", combined_text),
        "confirmation_id_count": _count_sentences_with(r"\bconfirmation\b|\btracking\b|\breceipt\b|\bid\b", combined_text),
        "contingency_count": _count_sentences_with(r"\bif .* then\b|\bfallback\b|\bbackup\b|\botherwise\b", combined_text),
        "prerequisite_count": _count_sentences_with(r"\bbefore you start\b|\bprerequisite\b|\byou will need\b", combined_text),
        "next_step_count": _count_sentences_with(r"\bnext step\b|\bready for\b|\bremaining\b", combined_text),
        "audience_term_hits": sum(normalized_text.count(term) for term in audience_terms),
        "must_include_hits": sum(normalized_text.count(term) for term in must_include_terms),
    }

    for section_name, patterns in SECTION_PATTERNS.items():
        matches = _section_matches(sections, patterns)
        feature_map[f"has_{section_name}_section"] = bool(matches)
        feature_map[f"{section_name}_section_count"] = len(matches)

    feature_map["required_section_coverage"] = 0.0
    if required_sections:
        hit_count = sum(1 for section in required_sections if any(section in heading for heading in sections))
        feature_map["required_section_coverage"] = hit_count / len(required_sections)

    return feature_map


def _iter_dict_nodes(value: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if isinstance(value, dict):
        nodes.append(value)
        for nested in value.values():
            nodes.extend(_iter_dict_nodes(nested))
    elif isinstance(value, list):
        for item in value:
            nodes.extend(_iter_dict_nodes(item))
    return nodes


def extract_trace_features(traces: list[TraceInput]) -> dict[str, Any]:
    features: Counter[str] = Counter()
    features["trace_count"] = len(traces)
    event_nodes: list[dict[str, Any]] = []

    for trace in traces:
        features["trace_text_length"] += len(trace.text)
        if trace.parsed is not None:
            event_nodes.extend(_iter_dict_nodes(trace.parsed))
        features["trace_verification_mentions"] += trace.text.lower().count("verify")
        features["trace_approval_mentions"] += trace.text.lower().count("approval")
        features["trace_receipt_mentions"] += trace.text.lower().count("receipt")

    for node in event_nodes:
        fingerprint = " ".join(str(value).lower() for value in node.values())
        features["event_count"] += 1
        if any(key in node for key in ("type", "event", "phase", "status")):
            features["typed_event_count"] += 1
        if "search" in fingerprint:
            features["search_event_count"] += 1
        if "source" in fingerprint:
            features["source_event_count"] += 1
        if "option" in fingerprint or "alternative" in fingerprint:
            features["option_event_count"] += 1
        if "risk" in fingerprint or "edge" in fingerprint:
            features["risk_event_count"] += 1
        if "approval" in fingerprint or "approve" in fingerprint or "checkpoint" in fingerprint:
            features["approval_event_count"] += 1
        if "verify" in fingerprint or "receipt" in fingerprint or "confirmation" in fingerprint:
            features["verification_event_count"] += 1
        if "save" in fingerprint or "persist" in fingerprint or "write" in fingerprint:
            features["save_event_count"] += 1
        if "block" in fingerprint or "wait" in fingerprint:
            features["blocker_event_count"] += 1
        if "resume" in fingerprint:
            features["resume_event_count"] += 1
        if "revise" in fingerprint or "iteration" in fingerprint or "retry" in fingerprint:
            features["revision_event_count"] += 1
        if "test" in fingerprint:
            features["test_event_count"] += 1
        if "build" in fingerprint:
            features["build_event_count"] += 1
        if "lint" in fingerprint or "typecheck" in fingerprint:
            features["lint_event_count"] += 1
        if "deploy" in fingerprint:
            features["deploy_event_count"] += 1
        if "handoff" in fingerprint or "delegate" in fingerprint:
            features["handoff_event_count"] += 1
        if "error" in fingerprint or "fail" in fingerprint:
            features["error_event_count"] += 1
        if node.get("approved") is True:
            features["approved_count"] += 1
        if node.get("verified") is True:
            features["verified_count"] += 1
        if node.get("success") is True:
            features["success_count"] += 1
        if isinstance(node.get("coverage"), (int, float)):
            features["reported_coverage_values_count"] += 1
            features["reported_coverage_total"] += float(node["coverage"])

    if features["reported_coverage_values_count"]:
        features["reported_coverage_avg"] = (
            features["reported_coverage_total"] / features["reported_coverage_values_count"]
        )
    else:
        features["reported_coverage_avg"] = 0.0

    return dict(features)


def merge_feature_maps(*feature_maps: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for feature_map in feature_maps:
        for key, value in feature_map.items():
            if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
    return merged


def safe_mean(values: list[float]) -> float:
    clean = [value for value in values if value is not None]
    return fmean(clean) if clean else 0.0
