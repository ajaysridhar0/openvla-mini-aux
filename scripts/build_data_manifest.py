#!/usr/bin/env python3
"""Rebuild and checksum the BARX data manifest from a staged release tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Portable staged tree")
    parser.add_argument("--template", type=Path, default=Path("dataset/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/manifest.csv"))
    parser.add_argument("--hash-workers", type=int, default=4)
    args = parser.parse_args()
    if args.hash_workers < 1:
        parser.error("--hash-workers must be positive")

    with args.template.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths = [args.data_root.resolve() / row["relative_path"] for row in rows]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Portable release is incomplete: {len(missing)} manifest files are missing")

    with ThreadPoolExecutor(max_workers=args.hash_workers) as executor:
        hashes = list(executor.map(sha256, paths))
    for row, path, digest in zip(rows, paths, hashes):
        row["bytes"] = str(path.stat().st_size)
        row["sha256"] = digest

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = args.output.with_suffix(args.output.suffix + ".staging")
    with staging.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(staging, args.output)
    print(f"Wrote {len(rows)} checksummed files to {args.output}")


if __name__ == "__main__":
    main()
