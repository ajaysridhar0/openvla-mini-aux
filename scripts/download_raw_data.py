#!/usr/bin/env python3
"""Download and verify one RLDS-aligned BARX raw HDF5 subset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "raw_dataset.json"
SUBSET_ROOT = ROOT / "dataset" / "subsets"
LOCAL_MANIFEST_NAME = "subset-manifest.csv"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from barx.raw_data import (  # noqa: E402
    PAPER_SET_NAMES,
    TARGET_NAMES,
    TASK_NAMES,
    RawSubset,
)
from scripts.download_public_artifacts import (  # noqa: E402
    DownloadSpec,
    _nonnegative_int,
    _positive_int,
    download_snapshot,
)
from scripts.verify_raw_data import verify_dataset  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=PAPER_SET_NAMES, required=True)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--target", choices=TARGET_NAMES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=_positive_int, default=1)
    parser.add_argument("--etag-timeout", type=_positive_int, default=60)
    parser.add_argument("--retries", type=_nonnegative_int, default=6)
    parser.add_argument("--verify-workers", type=_positive_int, default=4)
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def selection(args: argparse.Namespace) -> RawSubset:
    try:
        return RawSubset(args.dataset, args.task, args.target)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def subset_manifest(subset: RawSubset) -> Path:
    path = SUBSET_ROOT / f"{subset.slug}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing generated subset manifest: {path}. "
            "Run scripts/build_raw_subsets.py."
        )
    return path


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Subset manifest is empty: {path}")
    return rows


def download_spec(
    args: argparse.Namespace, subset: RawSubset
) -> tuple[DownloadSpec, Path, list[dict[str, str]]]:
    config = json.loads(DATASET_CONFIG.read_text())
    manifest = subset_manifest(subset)
    rows = manifest_rows(manifest)
    patterns = ("README.md", "LICENSE", "manifest.csv") + tuple(
        row["relative_path"] for row in rows
    )
    return (
        DownloadSpec(
            name=subset.slug,
            repo_id=config["repo_id"],
            repo_type=config["repo_type"],
            revision=config["revision"],
            local_dir=args.output_dir.expanduser().resolve(),
            allow_patterns=patterns,
            expected_files={},
        ),
        manifest,
        rows,
    )


def main() -> None:
    args = build_parser().parse_args()
    subset = selection(args)
    spec, manifest, rows = download_spec(args, subset)
    size_bytes = sum(int(row["bytes"]) for row in rows)
    demonstrations = sum(int(row["demonstrations"]) for row in rows)
    print(
        f"{subset.slug}: {len(rows)} files, {demonstrations} demonstrations, "
        f"{size_bytes / 2**30:.2f} GiB"
    )
    print(
        f"{spec.repo_id}@{spec.revision} -> {spec.local_dir} "
        f"(RLDS: {subset.rlds_repo_id})"
    )
    if args.dry_run:
        for row in rows:
            print(row["relative_path"])
        return

    from huggingface_hub import snapshot_download

    download_snapshot(
        spec, spec.local_dir / ".hf-cache", args, snapshot_download
    )
    local_manifest = spec.local_dir / LOCAL_MANIFEST_NAME
    shutil.copyfile(manifest, local_manifest)
    files, verified_demos, verified_bytes = verify_dataset(
        spec.local_dir,
        local_manifest,
        workers=args.verify_workers,
        check_checksums=not args.skip_checksums,
    )
    checksum_status = "with SHA-256" if not args.skip_checksums else "without SHA-256"
    print(
        f"Verified {files} files, {verified_demos} demonstrations, "
        f"{verified_bytes / 2**30:.2f} GiB {checksum_status}"
    )


if __name__ == "__main__":
    main()
