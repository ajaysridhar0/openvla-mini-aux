#!/usr/bin/env python3
"""Verify a BARX raw HDF5 archive against the published manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "dataset" / "manifest.csv"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_data_manifest import sha256  # noqa: E402
from scripts.stage_release_data import verify_hdf5  # noqa: E402


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _verify_hdf5_path(item: tuple[Path, int]) -> None:
    verify_hdf5(*item)


def verify_inventory(
    data_root: Path, rows: list[dict[str, str]], workers: int
) -> list[Path]:
    paths = []
    for row in rows:
        path = data_root / row["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest file is missing: {path}")
        expected_bytes = int(row["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(
                f"Size mismatch for {path}: expected {expected_bytes}, found {path.stat().st_size}"
            )
        paths.append(path)
    verification_items = [
        (path, int(row["demonstrations"])) for path, row in zip(paths, rows)
    ]
    if workers == 1:
        for item in verification_items:
            _verify_hdf5_path(item)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_verify_hdf5_path, verification_items))
    return paths


def verify_checksums(
    paths: list[Path], rows: list[dict[str, str]], workers: int
) -> None:
    missing = [row["relative_path"] for row in rows if len(row.get("sha256", "")) != 64]
    if missing:
        raise ValueError(f"Manifest has {len(missing)} missing SHA-256 checksums")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        actual_hashes = list(executor.map(sha256, paths))
    mismatches = [
        row["relative_path"]
        for row, actual in zip(rows, actual_hashes)
        if actual != row["sha256"]
    ]
    if mismatches:
        raise ValueError(
            f"SHA-256 mismatch for {len(mismatches)} files: {', '.join(mismatches[:5])}"
        )


def verify_dataset(
    data_root: Path,
    manifest: Path = DEFAULT_MANIFEST,
    *,
    workers: int = 4,
    check_checksums: bool = True,
) -> tuple[int, int, int]:
    if workers < 1:
        raise ValueError("workers must be positive")
    rows = manifest_rows(manifest)
    paths = verify_inventory(data_root.resolve(), rows, workers)
    if check_checksums:
        verify_checksums(paths, rows, workers)
    return (
        len(rows),
        sum(int(row["demonstrations"]) for row in rows),
        sum(path.stat().st_size for path in paths),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root containing the manifest-relative human/ and mg/ trees",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--workers", type=int, default=4, help="parallel SHA-256 workers (default: 4)"
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="verify file sizes and HDF5 structure without reading all bytes",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    files, demonstrations, size_bytes = verify_dataset(
        args.data_root,
        args.manifest,
        workers=args.workers,
        check_checksums=not args.skip_checksums,
    )
    checksum_status = (
        "sizes, structure, and SHA-256"
        if not args.skip_checksums
        else "sizes and structure"
    )
    print(
        f"Verified {files} HDF5 files, {demonstrations} demonstrations, and "
        f"{size_bytes / 2**30:.2f} GiB ({checksum_status})"
    )


if __name__ == "__main__":
    main()
