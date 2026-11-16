from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dw_eval.adapters import evaluate_work_type
from dw_eval.io import load_artifacts, load_context, load_traces
from dw_eval.reporting import render_markdown_report
from dw_eval.shared import extract_artifact_features, extract_trace_features


ARTIFACT_FIXTURES = {
    "research": """# Executive Summary

## Key Findings
- Fact one with citation https://example.com/source
- Fact two with citation https://gov.example.gov/report

## Analysis
We evaluated multiple options and surfaced one key risk.

## Recommendations
1. Owner: Pritesh. By April 1, launch the pilot.

## Sources
- https://example.com/source
- https://gov.example.gov/report
""",
    "design": """# Problem Statement
## Requirements
- Must support OAuth
## Constraints
- Must work with existing browser sessions
## Options Evaluated
- Option A
- Option B
## Decision Matrix
| Criteria | Weight |
| --- | --- |
## Implementation Roadmap
- Owner: Team
## Risks & Mitigations
- Risk: login drift
""",
    "content": """# Executive Summary
Please review this note and reply by Monday.

## Next Steps
- Let me know if you approve the rollout.
""",
    "software": """# Implementation
- Added test coverage and error handling.

## Verification
- Tests passed
""",
    "outreach": """# Outreach Package

Please confirm whether Tuesday 2pm or Wednesday 11am works.

## Follow-Up Plan
- If no reply by Friday, send a follow-up.
""",
    "decomposition": """# Breakdown
- Task 1
- Task 2

## Dependencies
- Task 2 depends on Task 1

## Acceptance Criteria
- Done when deployed
""",
    "procedure": """# Procedure

## Prerequisites
- Account access

## Steps
1. Open https://portal.example.com
2. Call (555) 555-1212 if login fails

## Verification
- Record confirmation ID

## Contingencies
- If blocked, use the fallback form
""",
    "operations": """# Receipt

## Verification
- Confirmation ID: ABC-123
- Result URL: https://example.com/ticket/ABC-123

## Remaining Follow-Up
- None
""",
}


TRACE_FIXTURE = {
    "events": [
        {"type": "search", "status": "success"},
        {"type": "source", "status": "success", "verified": True},
        {"type": "verification", "status": "success", "verified": True},
        {"type": "save", "status": "success"},
        {"type": "approval", "status": "success", "approved": True},
        {"type": "revision", "status": "success"},
        {"type": "test", "status": "success"},
        {"type": "build", "status": "success"},
        {"type": "lint", "status": "success"},
    ]
}


class DwEvalTestCase(unittest.TestCase):
    def _evaluate(self, work_type: str, text: str, trace: dict | None = None, context: dict | None = None):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / f"{work_type}.md"
            artifact_path.write_text(text, encoding="utf-8")
            trace_paths: list[str] = []
            if trace is not None:
                trace_path = root / f"{work_type}-trace.json"
                trace_path.write_text(json.dumps(trace), encoding="utf-8")
                trace_paths.append(str(trace_path))

            artifacts = load_artifacts([str(artifact_path)])
            traces = load_traces(trace_paths)
            artifact_features = extract_artifact_features(artifacts, context or {})
            trace_features = extract_trace_features(traces)
            return evaluate_work_type(
                work_type=work_type,
                title=work_type,
                artifact_refs=[str(artifact_path)],
                process_trace_refs=trace_paths,
                artifact_features=artifact_features,
                trace_features=trace_features,
                context=context or {},
            )

    def test_all_work_types_score(self):
        for work_type, text in ARTIFACT_FIXTURES.items():
            with self.subTest(work_type=work_type):
                result = self._evaluate(work_type, text, TRACE_FIXTURE)
                self.assertGreater(result.overall_score, 0)
                self.assertEqual(set(result.artifact.dimensions.keys()), {"coverage", "correctness", "specificity", "verification", "robustness", "closure"})
                report = render_markdown_report(result)
                self.assertIn("Evaluation Report", report)

    def test_software_requires_test_evidence(self):
        result = self._evaluate("software", ARTIFACT_FIXTURES["software"], trace={"events": [{"type": "save"}]})
        self.assertTrue(any("test evidence" in gate.lower() for gate in result.gate_failures))

    def test_operations_requires_approval_for_risky_action(self):
        result = self._evaluate(
            "operations",
            ARTIFACT_FIXTURES["operations"],
            trace={"events": [{"type": "verification"}]},
            context={"action_class": "send", "zone": "personal", "approval_required": True},
        )
        self.assertTrue(any("approval checkpoint" in gate.lower() for gate in result.gate_failures))

    def test_execution_mode_requires_next_human_action_or_questions(self):
        result = self._evaluate(
            "content",
            "# Draft\n\nThis is a short draft without closure.",
            trace={"events": [{"type": "save"}]},
            context={"execution_mode": "Coordinate"},
        )
        self.assertTrue(any("next human action" in gate.lower() for gate in result.gate_failures))

    def test_declared_ambiguity_requires_clarifying_questions(self):
        result = self._evaluate(
            "decomposition",
            """# Breakdown

## Clarifying Questions
- What outcome would make this done?
""",
            trace={"events": [{"type": "save"}]},
            context={"execution_mode": "Decide", "requires_clarification": True},
        )
        self.assertFalse(any("blocking ambiguity" in gate.lower() for gate in result.gate_failures))


class JsonlTraceTests(unittest.TestCase):
    def test_jsonl_trace_events_are_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text('{"type": "test", "status": "passed"}\n{"type": "build", "status": "ok"}\n', encoding="utf-8")
            features = extract_trace_features(load_traces([str(path)]))
        self.assertEqual(features["test_event_count"], 1)
        self.assertEqual(features["build_event_count"], 1)


if __name__ == "__main__":
    unittest.main()
