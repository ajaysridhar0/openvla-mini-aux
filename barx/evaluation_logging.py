"""Durable local logging for BARX evaluation runs."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


def json_ready(value: Any) -> Any:
    """Convert dataclasses, paths, and nested containers to JSON values."""

    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON without exposing a partially written destination file."""

    staging = path.with_suffix(path.suffix + ".staging")
    staging.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, path)


def checkpoint_label(checkpoint: str | Path) -> str:
    """Return a readable filesystem-safe label for a checkpoint."""

    checkpoint = str(checkpoint)
    match = re.search(r"step-(\d+)(?:-epoch)?", checkpoint)
    if match:
        step = int(match.group(1))
        return f"step-{step}"
    stem = Path(checkpoint).stem or "checkpoint"
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return label or "checkpoint"


def create_run_directory(template: str | Path, checkpoint: str | Path) -> Path:
    """Create a non-overwriting run directory from a public launcher template."""

    base = Path(str(template).replace("STEP", checkpoint_label(checkpoint)))
    for attempt in range(1, 10_000):
        candidate = base if attempt == 1 else Path(f"{base}-run-{attempt:03d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not allocate a run directory under {base.parent}")


class EvaluationRun:
    """Own the durable files produced by one evaluation invocation."""

    def __init__(self, directory: Path, config: Any):
        self.directory = directory
        self.started_at = datetime.now(timezone.utc)
        self.log_path = directory / "log.txt"
        self.episodes_path = directory / "episodes.jsonl"
        self.summary_path = directory / "summary.json"
        self.log_file: TextIO = self.log_path.open("x", encoding="utf-8")
        self._episodes: TextIO = self.episodes_path.open("x", encoding="utf-8")
        self.episode_count = 0
        self.success_count = 0
        atomic_write_json(directory / "config.json", config)

    def write(self, message: str) -> None:
        self.log_file.write(message)
        self.log_file.flush()

    def record_episode(self, record: dict[str, Any]) -> None:
        self._episodes.write(json.dumps(json_ready(record), sort_keys=True) + "\n")
        self._episodes.flush()
        self.episode_count += 1
        self.success_count += int(bool(record.get("success")))

    def finalize(
        self,
        summary: dict[str, Any],
        *,
        status: str,
        error: BaseException | None = None,
    ) -> None:
        finished_at = datetime.now(timezone.utc)
        payload = {
            **summary,
            "status": status,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - self.started_at).total_seconds(),
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)
        atomic_write_json(self.summary_path, payload)

    def close(self) -> None:
        if not self._episodes.closed:
            self._episodes.close()
        if not self.log_file.closed:
            self.log_file.close()

    def __enter__(self) -> EvaluationRun:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
