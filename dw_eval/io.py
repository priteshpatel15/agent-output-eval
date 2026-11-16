from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import EvaluationResult


TEXT_FILE_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".html",
    ".css",
    ".sql",
}


@dataclass
class ArtifactInput:
    path: Path
    text: str
    file_count: int = 1


@dataclass
class TraceInput:
    path: Path
    text: str
    parsed: Any | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:40] or "untitled"


def deep_work_root() -> Path:
    return Path(os.environ.get("DEEP_WORK_ROOT", str(Path.home() / "deep-work-system")))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_directory(path: Path) -> ArtifactInput:
    parts: list[str] = []
    file_count = 0
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in TEXT_FILE_SUFFIXES:
            continue
        file_count += 1
        parts.append(f"\n\n=== FILE: {candidate.relative_to(path)} ===\n")
        parts.append(_read_text_file(candidate))
    return ArtifactInput(path=path, text="".join(parts), file_count=file_count)


def load_artifacts(paths: list[str]) -> list[ArtifactInput]:
    artifacts: list[ArtifactInput] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            artifacts.append(_load_directory(path))
            continue
        artifacts.append(ArtifactInput(path=path, text=_read_text_file(path)))
    return artifacts


def load_traces(paths: list[str]) -> list[TraceInput]:
    traces: list[TraceInput] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        text = _read_text_file(path)
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        traces.append(TraceInput(path=path, text=text, parsed=parsed))
    return traces


def load_context(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    context_path = Path(path).expanduser().resolve()
    text = _read_text_file(context_path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Context file must be valid JSON: {context_path}") from exc


def default_save_paths(work_type: str, title: str) -> dict[str, Path]:
    root = deep_work_root()
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(title)
    json_path = root / "context" / "history" / "evaluations" / work_type / f"{date_str}-{slug}.json"
    md_path = root / "outputs" / "quality-reports" / work_type / f"{date_str}-{slug}.md"
    return {"json": json_path, "markdown": md_path}


def save_evaluation(result: EvaluationResult, markdown: str, json_path: Path, markdown_path: Path) -> dict[str, str]:
    ensure_parent(json_path)
    ensure_parent(markdown_path)
    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def dump_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)
