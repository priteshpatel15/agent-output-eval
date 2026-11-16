# agent-output-eval

**A deterministic scorer for what an agent delivers and for the trace of how it got there.**

An agent's report says "all tests pass." Did any test run? This tool does not ask a model to grade the report. It counts evidence: test events, build events, citations, approval checkpoints, saved files. A deliverable that claims verification without evidence in the artifact or the trace fails a gate.

Author: Pritesh Patel

## Quickstart

Requires Python 3.10+. No dependencies.

```bash
python3 -m unittest discover -s tests
python3 -m dw_eval.cli evaluate --work-type software --artifact report.md --trace run.jsonl --print-report
```

A software report that says "All tests pass" with no trace scores 1.75/5 and returns:

```
## Gate Failures
- Software evaluation is missing explicit test evidence.
```

## How it works

- `dw_eval/shared.py` extracts countable features from artifacts (headings, citations, checklists) and from JSON or JSONL traces (typed events, test, build, verification, approval and save events).
- `dw_eval/adapters.py` holds one adapter per work type: research, design, content, software, outreach, decomposition, procedure, operations. Each maps features to six dimensions and to hard gates.
- `dw_eval/scoring.py` turns feature ratios into 1 to 5 scores with a confidence value.
- `dw_eval/reporting.py` renders a markdown report. `dw_eval/cli.py` is the entry point (`evaluate`, `extract`, `report`).

Every result has two scorecards. The artifact scorecard grades the deliverable. The process scorecard grades the trace. Both use the same six dimensions: coverage, correctness, specificity, verification, robustness, closure.

## What it does not judge

Some qualities are not countable. The scorer lists them under "residual judgments" (for example tone in outreach) and leaves them to a person. It does not fake a number for them.

## Optional model-assisted extraction

`--use-ai` asks an OpenAI model to count semantic features (claims, risks, edge cases). Scoring stays deterministic. Needs `OPENAI_API_KEY` and `DW_EVAL_OPENAI_MODEL`. Off by default.

## Limits

- Features are keyword and structure counts. An agent that writes fake test events into its trace would pass. The trace must come from the runtime, not from the agent.
- Thresholds were set by hand against my own task history, not fit to labeled data.
- `--save` writes under `$DEEP_WORK_ROOT` (default `~/deep-work-system`).

## License

MIT
