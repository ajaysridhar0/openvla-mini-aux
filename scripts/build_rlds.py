#!/usr/bin/env python3
"""Build one training-compatible BARX RLDS dataset from the release manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset" / "manifest.csv"
PAPER_SET_NAMES = {
    "xp_900": "XP-900",
    "xp_3k": "XP-3K",
    "sp_900": "SP-900",
    "target_50": "target-50",
}
TASK_NAMES = {
    "pnp": {
        "pick-and-place counter-to-sink",
        "pick-and-place sink-to-counter",
    },
    "turn_on_sink": {"turn-on-sink-faucet"},
    "flip_mug": {"flip-mug-upright"},
}
TARGET_NAMES = {
    "panda": "Panda",
    "panda_og": "Panda-OG",
    "jaco": "Jaco",
}


def stored_dataset_name(dataset: str, task: str, target: str | None) -> str:
    """Return the original TFDS name retained for checkpoint compatibility."""
    if dataset == "xp_900":
        return f"mg_{task}_lite"
    if dataset == "xp_3k":
        return f"mg_{task}"
    if target is None:
        raise ValueError(f"{dataset} requires a target embodiment")
    if dataset == "sp_900":
        return f"mg_{target}_{task}"
    return f"{target}_{task}"


def selected_paths(dataset: str, task: str, target: str | None, raw_root: Path) -> list[Path]:
    paper_set = PAPER_SET_NAMES[dataset]
    target_name = TARGET_NAMES.get(target) if target else None
    selected = []
    with MANIFEST.open(newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            memberships = set(row["paper_sets"].split(";"))
            if paper_set not in memberships or row["task"] not in TASK_NAMES[task]:
                continue
            if target_name is not None and row["embodiment"] != target_name:
                continue
            path = raw_root / row["relative_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Manifest file is missing: {path}")
            if path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"Size mismatch for {path}")
            selected.append(path.resolve())
    if not selected:
        raise ValueError(f"No manifest entries matched {dataset}/{target or 'source'}/{task}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=PAPER_SET_NAMES, required=True)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--target", choices=TARGET_NAMES)
    parser.add_argument("--raw-root", type=Path, required=True, help="Directory containing manifest-relative HDF5 files")
    parser.add_argument("--rlds-root", type=Path, required=True, help="TFDS output root used by policy training")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--path-batch", type=int, default=160)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requires_target = args.dataset in {"sp_900", "target_50"}
    if requires_target and args.target is None:
        parser.error(f"{args.dataset} requires --target")
    if not requires_target and args.target is not None:
        parser.error(f"{args.dataset} does not accept --target")
    if args.workers < 1 or args.path_batch < 1:
        parser.error("--workers and --path-batch must be positive")

    output_name = stored_dataset_name(args.dataset, args.task, args.target)
    paths = selected_paths(args.dataset, args.task, args.target, args.raw_root)
    print(f"{args.dataset}/{args.target or 'source'}/{args.task}: {len(paths)} files -> {args.rlds_root / output_name}")
    if args.dry_run:
        return

    os.environ["BARX_RAW_DATA_GLOB"] = os.pathsep.join(map(str, paths))
    os.environ["BARX_CONVERTER_WORKERS"] = str(args.workers)
    os.environ["BARX_CONVERTER_PATH_BATCH"] = str(args.path_batch)

    # TFDS derives the on-disk dataset name from the builder class name. A
    # dynamic thin subclass gives each logical paper split its historical name
    # while all splits share exactly one conversion implementation.
    from dataset.rlds.robocasa_x_dataset_builder import RobocasaXDataset

    class_name = "".join(part.capitalize() for part in output_name.split("_"))
    builder_type = type(class_name, (RobocasaXDataset,), {"__module__": __name__})
    builder = builder_type(data_dir=str(args.rlds_root))
    if builder.name != output_name:
        raise RuntimeError(f"TFDS derived {builder.name!r}; expected {output_name!r}")
    builder.download_and_prepare()
    print(f"Prepared {builder.info.full_name} at {builder.data_dir}")


if __name__ == "__main__":
    main()
