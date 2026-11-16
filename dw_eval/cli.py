from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import evaluate_work_type
from .ai import extract_semantic_features
from .io import (
    default_save_paths,
    dump_json,
    load_artifacts,
    load_context,
    load_traces,
    save_evaluation,
    slugify,
)
from .models import WORK_TYPES
from .reporting import render_markdown_report
from .shared import extract_artifact_features, extract_trace_features, merge_feature_maps


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dw-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--work-type", required=True, choices=WORK_TYPES)
    evaluate.add_argument("--artifact", action="append", required=True)
    evaluate.add_argument("--trace", action="append", default=[])
    evaluate.add_argument("--context")
    evaluate.add_argument("--title")
    evaluate.add_argument("--zone")
    evaluate.add_argument("--system")
    evaluate.add_argument("--board-category")
    evaluate.add_argument("--execution-mode")
    evaluate.add_argument("--action-class")
    evaluate.add_argument("--approval-required", action="store_true")
    evaluate.add_argument("--requires-clarification", action="store_true")
    evaluate.add_argument("--use-ai", action="store_true")
    evaluate.add_argument("--save", action="store_true", default=False)
    evaluate.add_argument("--print-report", action="store_true")

    report = subparsers.add_parser("report")
    report.add_argument("evaluation_json")
    report.add_argument("--write")

    extract = subparsers.add_parser("extract")
    extract.add_argument("--work-type", choices=WORK_TYPES)
    extract.add_argument("--artifact", action="append", required=True)
    extract.add_argument("--trace", action="append", default=[])
    extract.add_argument("--context")
    extract.add_argument("--use-ai", action="store_true")

    return parser


def _resolve_title(args: argparse.Namespace, artifact_paths: list[str], context: dict[str, object]) -> str:
    if args.title:
        return args.title
    if context.get("title"):
        return str(context["title"])
    return Path(artifact_paths[0]).stem.replace("-", " ")


def _run_evaluate(args: argparse.Namespace) -> int:
    context = load_context(args.context)
    if args.zone:
        context.setdefault("zone", args.zone)
    if args.system:
        context.setdefault("system", args.system)
    if args.board_category:
        context.setdefault("board_category", args.board_category)
    if args.execution_mode:
        context.setdefault("execution_mode", args.execution_mode)
    if args.action_class:
        context.setdefault("action_class", args.action_class)
    if args.approval_required:
        context["approval_required"] = True
    if args.requires_clarification:
        context["requires_clarification"] = True

    artifacts = load_artifacts(args.artifact)
    traces = load_traces(args.trace)
    artifact_features = extract_artifact_features(artifacts, context)
    trace_features = extract_trace_features(traces)

    if args.use_ai:
        semantic = extract_semantic_features(
            args.work_type,
            "\n\n".join(artifact.text for artifact in artifacts),
            context,
        )
        artifact_features = merge_feature_maps(artifact_features, semantic)

    title = _resolve_title(args, args.artifact, context)
    result = evaluate_work_type(
        work_type=args.work_type,
        title=title,
        artifact_refs=[str(artifact.path) for artifact in artifacts],
        process_trace_refs=[str(trace.path) for trace in traces],
        artifact_features=artifact_features,
        trace_features=trace_features,
        context=context,
    )

    report_markdown = render_markdown_report(result)
    if args.save:
        save_paths = default_save_paths(args.work_type, title)
        result.saved_paths = save_evaluation(result, report_markdown, save_paths["json"], save_paths["markdown"])

    payload = result.to_dict()
    if args.print_report:
        print(report_markdown)
    else:
        print(dump_json(payload))
    return 0


def _run_report(args: argparse.Namespace) -> int:
    raw = Path(args.evaluation_json).expanduser().resolve().read_text(encoding="utf-8")
    payload = json.loads(raw)
    report = render_markdown_report(_payload_to_result(payload))
    if args.write:
        output_path = Path(args.write).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0


def _run_extract(args: argparse.Namespace) -> int:
    context = load_context(args.context)
    artifacts = load_artifacts(args.artifact)
    traces = load_traces(args.trace)
    artifact_features = extract_artifact_features(artifacts, context)
    trace_features = extract_trace_features(traces)
    payload = {"artifact_features": artifact_features, "trace_features": trace_features}
    if args.use_ai:
        payload["semantic_features"] = extract_semantic_features(
            args.work_type or "research",
            "\n\n".join(artifact.text for artifact in artifacts),
            context,
        )
    print(dump_json(payload))
    return 0


def _payload_to_result(payload: dict) -> object:
    from .models import DimensionScore, EvaluationResult, ScoreBlock

    def to_block(block_payload: dict) -> ScoreBlock:
        dimensions = {
            name: DimensionScore(**dimension_payload)
            for name, dimension_payload in block_payload["dimensions"].items()
        }
        return ScoreBlock(
            label=block_payload["label"],
            dimensions=dimensions,
            average=block_payload["average"],
            confidence=block_payload["confidence"],
        )

    return EvaluationResult(
        title=payload["title"],
        work_type=payload["work_type"],
        generated_at=payload["generated_at"],
        artifact_refs=payload["artifact_refs"],
        process_trace_refs=payload["process_trace_refs"],
        artifact=to_block(payload["artifact"]),
        process=to_block(payload["process"]),
        overall_score=payload["overall_score"],
        overall_confidence=payload["overall_confidence"],
        gate_failures=payload.get("gate_failures", []),
        residual_judgments=payload.get("residual_judgments", []),
        features=payload.get("features", {}),
        metadata=payload.get("metadata", {}),
        saved_paths=payload.get("saved_paths", {}),
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "evaluate":
        return _run_evaluate(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "extract":
        return _run_extract(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
