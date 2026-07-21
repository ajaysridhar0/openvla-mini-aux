#!/usr/bin/env python3
"""Download the public artifacts for the smallest BARX training walkthrough."""

from __future__ import annotations

import argparse
from pathlib import Path

BASE_VLM_REPO = "ajaysri/prism-qwen25-extra-dinosiglip-224px-0_5b-stage-finetune-x7"
XP900_PNP_DATA_REPO = "ajaysri/robocasa-x-xp900-pnp"
XP900_PNP_VQ_REPO = "ajaysri/robocasa-x-xp900-pnp-vq-extra-action-tokenizer"
XP900_PNP_PRETRAIN_REPO = "ajaysri/barx-joint-reps-xp900-pnp-pretrain"


def downloads(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    items = [
        (BASE_VLM_REPO, "model", args.base_vlm_dir),
        (XP900_PNP_DATA_REPO, "dataset", args.data_root / "mg_pnp_lite"),
        (XP900_PNP_VQ_REPO, "model", args.vq_root / "mg_pnp_lite"),
    ]
    if args.include_pretrain_checkpoint:
        items.append((XP900_PNP_PRETRAIN_REPO, "model", args.run_root / "xp900-pnp-joint-reps"))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-vlm-dir", type=Path, required=True)
    parser.add_argument("--vq-root", type=Path, default=Path("policy/vq"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/public"))
    parser.add_argument("--include-pretrain-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for repo_id, repo_type, local_dir in downloads(args):
        local_dir = local_dir.expanduser().resolve()
        print(f"{repo_type:7} {repo_id} -> {local_dir}")
        if not args.dry_run:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=local_dir,
                token=False,
            )


if __name__ == "__main__":
    main()
