#!/usr/bin/env python3
"""Run one acceptance-test script and preserve exact command evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_stream(source: BinaryIO, output: BinaryIO, terminal: BinaryIO) -> None:
    for block in iter(lambda: source.read(64 * 1024), b""):
        output.write(block)
        output.flush()
        terminal.write(block)
        terminal.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--command-id", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--section", required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    args = parser.parse_args()

    if args.attempt < 1:
        parser.error("--attempt must be positive")
    cwd = args.cwd.resolve()
    source_script = args.script.resolve()
    if not cwd.is_dir():
        parser.error(f"--cwd is not a directory: {cwd}")
    if not source_script.is_file():
        parser.error(f"--script is not a file: {source_script}")

    command_dir = args.run_dir / "commands"
    log_dir = args.run_dir / "logs"
    command_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.command_id}-A{args.attempt:02d}"
    saved_script = command_dir / f"{stem}.sh"
    stdout_path = log_dir / f"{stem}.stdout.txt"
    stderr_path = log_dir / f"{stem}.stderr.txt"
    if any(path.exists() for path in (saved_script, stdout_path, stderr_path)):
        parser.error(f"evidence already exists for {stem}")
    shutil.copyfile(source_script, saved_script)

    started_at = utc_now()
    start = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            ["bash", str(saved_script)],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        threads = [
            threading.Thread(
                target=copy_stream,
                args=(process.stdout, stdout, sys.stdout.buffer),
            ),
            threading.Thread(
                target=copy_stream,
                args=(process.stderr, stderr, sys.stderr.buffer),
            ),
        ]
        for thread in threads:
            thread.start()
        exit_code = process.wait()
        for thread in threads:
            thread.join()
    ended_at = utc_now()

    record = {
        "command_id": args.command_id,
        "attempt": args.attempt,
        "section": args.section,
        "cwd": str(cwd),
        "command_file": saved_script.relative_to(args.run_dir).as_posix(),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(time.monotonic() - start, 3),
        "exit_code": exit_code,
        "status": "PASS" if exit_code == 0 else "FAIL",
        "stdout": stdout_path.relative_to(args.run_dir).as_posix(),
        "stderr": stderr_path.relative_to(args.run_dir).as_posix(),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }
    with (args.run_dir / "commands.jsonl").open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
