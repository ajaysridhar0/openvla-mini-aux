#!/usr/bin/env python3
"""
Batch stage and upload the RoboCasa-X TFDS datasets with renamed public-facing TFDS builder ids.

Each public Hugging Face dataset repo uses a paper-facing slug, e.g.
`ajaysri/robocasa-x-xp900-flip-mug-upright`, while the TFDS dataset payload inside the repo is
rewritten to an underscore-safe builder id such as `robocasa_x_xp900_flip_mug_upright`.

The script processes one dataset at a time to limit local disk usage:
1. stage source TFDS export under the renamed builder id
2. delete/recreate the target HF dataset repo if requested
3. upload the staged folder
4. optionally remove the staged folder
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from huggingface_hub import HfApi


ROBOCASA_X_HF_DATASET_RELEASES: List[Dict[str, str]] = [
    {
        "source_name": "mg_pnp",
        "target_name": "robocasa_x_xp3k_pnp",
        "repo_id": "ajaysri/robocasa-x-xp3k-pnp",
    },
    {
        "source_name": "mg_pnp_lite",
        "target_name": "robocasa_x_xp900_pnp",
        "repo_id": "ajaysri/robocasa-x-xp900-pnp",
    },
    {
        "source_name": "mg_turn_on_sink",
        "target_name": "robocasa_x_xp3k_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-xp3k-turn-on-sink-faucet",
    },
    {
        "source_name": "mg_turn_on_sink_lite",
        "target_name": "robocasa_x_xp900_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-xp900-turn-on-sink-faucet",
    },
    {
        "source_name": "mg_flip_mug",
        "target_name": "robocasa_x_xp3k_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-xp3k-flip-mug-upright",
    },
    {
        "source_name": "mg_flip_mug_lite",
        "target_name": "robocasa_x_xp900_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-xp900-flip-mug-upright",
    },
    {
        "source_name": "mg_panda_pnp",
        "target_name": "robocasa_x_sp900_panda_pnp",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-pnp",
    },
    {
        "source_name": "mg_panda_og_pnp",
        "target_name": "robocasa_x_sp900_panda_og_pnp",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-og-pnp",
    },
    {
        "source_name": "mg_jaco_pnp",
        "target_name": "robocasa_x_sp900_jaco_pnp",
        "repo_id": "ajaysri/robocasa-x-sp900-jaco-pnp",
    },
    {
        "source_name": "mg_panda_turn_on_sink",
        "target_name": "robocasa_x_sp900_panda_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-turn-on-sink-faucet",
    },
    {
        "source_name": "mg_panda_og_turn_on_sink",
        "target_name": "robocasa_x_sp900_panda_og_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-og-turn-on-sink-faucet",
    },
    {
        "source_name": "mg_jaco_turn_on_sink",
        "target_name": "robocasa_x_sp900_jaco_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-sp900-jaco-turn-on-sink-faucet",
    },
    {
        "source_name": "mg_panda_flip_mug",
        "target_name": "robocasa_x_sp900_panda_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-flip-mug-upright",
    },
    {
        "source_name": "mg_panda_og_flip_mug",
        "target_name": "robocasa_x_sp900_panda_og_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-og-flip-mug-upright",
    },
    {
        "source_name": "mg_jaco_flip_mug",
        "target_name": "robocasa_x_sp900_jaco_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-sp900-jaco-flip-mug-upright",
    },
    {
        "source_name": "panda_pnp",
        "target_name": "robocasa_x_target_panda_pnp",
        "repo_id": "ajaysri/robocasa-x-target-panda-pnp",
    },
    {
        "source_name": "panda_og_pnp",
        "target_name": "robocasa_x_target_panda_og_pnp",
        "repo_id": "ajaysri/robocasa-x-target-panda-og-pnp",
    },
    {
        "source_name": "jaco_pnp",
        "target_name": "robocasa_x_target_jaco_pnp",
        "repo_id": "ajaysri/robocasa-x-target-jaco-pnp",
    },
    {
        "source_name": "panda_turn_on_sink",
        "target_name": "robocasa_x_target_panda_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-target-panda-turn-on-sink-faucet",
    },
    {
        "source_name": "panda_og_turn_on_sink",
        "target_name": "robocasa_x_target_panda_og_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-target-panda-og-turn-on-sink-faucet",
    },
    {
        "source_name": "jaco_turn_on_sink",
        "target_name": "robocasa_x_target_jaco_turn_on_sink_faucet",
        "repo_id": "ajaysri/robocasa-x-target-jaco-turn-on-sink-faucet",
    },
    {
        "source_name": "panda_flip_mug",
        "target_name": "robocasa_x_target_panda_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-target-panda-flip-mug-upright",
    },
    {
        "source_name": "panda_og_flip_mug",
        "target_name": "robocasa_x_target_panda_og_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-target-panda-og-flip-mug-upright",
    },
    {
        "source_name": "jaco_flip_mug",
        "target_name": "robocasa_x_target_jaco_flip_mug_upright",
        "repo_id": "ajaysri/robocasa-x-target-jaco-flip-mug-upright",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/iliad/u/jenseng/tensorflow_datasets"),
        help="Root directory containing the original TFDS dataset exports.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/iliad2/u/ajaysri/barx_hf_stage"),
        help="Root directory for staging one renamed dataset at a time.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of source dataset names to process, e.g. mg_pnp mg_pnp_lite.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Stage datasets locally but do not upload them to HF.",
    )
    parser.add_argument(
        "--delete-first",
        action="store_true",
        help="Delete and recreate the target HF dataset repo before uploading.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Recreate HF repos as private when used with --delete-first or on first create.",
    )
    parser.add_argument(
        "--keep-staged",
        action="store_true",
        help="Keep the staged dataset folder after each upload.",
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=8,
        help="Number of workers for HF upload_large_folder.",
    )
    return parser.parse_args()


def filtered_releases(only: List[str]) -> List[Dict[str, str]]:
    if not only:
        return ROBOCASA_X_HF_DATASET_RELEASES
    wanted = set(only)
    return [release for release in ROBOCASA_X_HF_DATASET_RELEASES if release["source_name"] in wanted]


def stage_dataset(release: Dict[str, str], source_root: Path, stage_root: Path) -> Path:
    script_path = Path(__file__).with_name("stage_barx_hf_tfds_dataset.py")
    source_dir = source_root / release["source_name"]
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--source-dir",
            str(source_dir),
            "--target-name",
            release["target_name"],
            "--stage-root",
            str(stage_root),
            "--overwrite",
        ],
        check=True,
    )
    return stage_root / release["target_name"]


def refresh_repo(api: HfApi, repo_id: str, private: bool, delete_first: bool) -> None:
    if delete_first:
        api.delete_repo(repo_id=repo_id, repo_type="dataset", missing_ok=True)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)


def upload_dataset(api: HfApi, repo_id: str, staged_dir: Path, num_workers: int) -> None:
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=staged_dir,
        num_workers=num_workers,
    )


def main() -> None:
    args = parse_args()
    releases = filtered_releases(args.only)
    api = HfApi()

    if not releases:
        raise ValueError("No RoboCasa-X dataset releases matched the requested --only filter.")

    for idx, release in enumerate(releases, start=1):
        print(
            f"[{idx}/{len(releases)}] {release['source_name']} -> "
            f"{release['target_name']} -> {release['repo_id']}",
            flush=True,
        )
        staged_dir = stage_dataset(release, args.source_root, args.stage_root)

        if not args.skip_upload:
            refresh_repo(api, release["repo_id"], private=args.private, delete_first=args.delete_first)
            upload_dataset(api, release["repo_id"], staged_dir, num_workers=args.upload_workers)

        if not args.keep_staged and staged_dir.exists():
            shutil.rmtree(staged_dir)

    print("Completed RoboCasa-X HF dataset batch staging/upload.", flush=True)


if __name__ == "__main__":
    main()
