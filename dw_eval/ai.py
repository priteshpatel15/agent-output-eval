from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def extract_semantic_features(
    work_type: str,
    artifact_text: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("DW_EVAL_OPENAI_MODEL")
    if not api_key or not model:
        raise SystemExit(
            "AI extraction requires OPENAI_API_KEY and DW_EVAL_OPENAI_MODEL to be set."
        )

    prompt = (
        "Extract semantic evaluation features for a deterministic scoring system. "
        "Return JSON only with numeric fields. "
        "Fields: semantic_claim_count, semantic_recommendation_count, "
        "semantic_counterargument_count, semantic_clear_ask_count, "
        "semantic_personalization_count, semantic_uncertainty_marker_count, "
        "semantic_edge_case_count, semantic_risk_count, semantic_option_count, "
        "semantic_acceptance_criteria_count, semantic_prerequisite_count, "
        "semantic_contingency_count, semantic_verification_step_count. "
        f"Work type: {work_type}. "
        f"Context keys: {sorted((context or {}).keys())}.\n\nArtifact:\n{artifact_text[:12000]}"
    )

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": "You are a strict schema extractor. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_RESPONSES_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"AI extraction request failed: {exc}") from exc

    text_chunks: list[str] = []
    for output in body.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text":
                text_chunks.append(content.get("text", ""))
    try:
        return json.loads("\n".join(text_chunks).strip())
    except json.JSONDecodeError as exc:
        raise SystemExit("AI extraction did not return valid JSON.") from exc
