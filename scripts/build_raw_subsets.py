#!/usr/bin/env python3
"""Build or check the 24 RLDS-aligned raw HDF5 subset manifests."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER_MANIFEST = ROOT / "dataset" / "manifest.csv"
SUBSET_ROOT = ROOT / "dataset" / "subsets"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from barx.raw_data import all_subsets, selected_rows  # noqa: E402


def render_subset(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    """Render a deterministic CSV subset."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def expected_manifests(master_manifest: Path = MASTER_MANIFEST) -> dict[Path, str]:
    """Return every expected subset path and complete CSV content."""

    with master_manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {master_manifest}")
        fieldnames = reader.fieldnames
    return {
        SUBSET_ROOT / f"{subset.slug}.csv": render_subset(
            fieldnames, list(selected_rows(rows, subset))
        )
        for subset in all_subsets()
    }


def build(*, check: bool = False) -> None:
    """Write all manifests, or fail if checked-in manifests are stale."""

    expected = expected_manifests()
    unexpected = (
        set(SUBSET_ROOT.glob("*.csv")) - set(expected) if SUBSET_ROOT.exists() else set()
    )
    if check:
        stale = [
            path
            for path, content in expected.items()
            if not path.is_file() or path.read_text() != content
        ]
        if stale or unexpected:
            names = ", ".join(path.name for path in sorted(set(stale) | unexpected))
            raise SystemExit(f"Raw subset manifests are stale: {names}")
        print(f"Verified {len(expected)} raw subset manifests")
        return

    SUBSET_ROOT.mkdir(parents=True, exist_ok=True)
    for path in unexpected:
        path.unlink()
    for path, content in expected.items():
        path.write_text(content)
    print(f"Wrote {len(expected)} raw subset manifests to {SUBSET_ROOT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build(check=args.check)


if __name__ == "__main__":
    main()
