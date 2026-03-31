#!/usr/bin/env python3
"""
Batch stage and upload the released RoboCasa-X VQ action tokenizers to Hugging Face.

Each tokenizer is uploaded as its own model repo with a RoboCasa-X public-facing slug, while
preserving the underlying checkpoint/config contents from the local `vq/` tree.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import HfApi


ROBOCASA_X_VQ_TOKENIZER_RELEASES: List[Dict[str, str]] = [
    {
        "source_dir": "mg_pnp",
        "repo_id": "ajaysri/robocasa-x-xp3k-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-xp3k-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X XP-3K PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda, Panda-OG, Jaco",
    },
    {
        "source_dir": "mg_pnp_lite",
        "repo_id": "ajaysri/robocasa-x-xp900-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-xp900-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X XP-900 PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda, Panda-OG, Jaco",
    },
    {
        "source_dir": "mg_turn_on_sink",
        "repo_id": "ajaysri/robocasa-x-xp3k-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-xp3k-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X XP-3K Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Panda, Panda-OG, Jaco",
    },
    {
        "source_dir": "mg_turn_on_sink_lite",
        "repo_id": "ajaysri/robocasa-x-xp900-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-xp900-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X XP-900 Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Panda, Panda-OG, Jaco",
    },
    {
        "source_dir": "mg_flip_mug",
        "repo_id": "ajaysri/robocasa-x-xp3k-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-xp3k-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X XP-3K Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Panda, Panda-OG, Jaco",
    },
    {
        "source_dir": "mg_flip_mug_lite",
        "repo_id": "ajaysri/robocasa-x-xp900-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-xp900-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X XP-900 Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Panda, Panda-OG, Jaco",
    },
    {
        "source_dir": "mg_panda_pnp",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-panda-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Panda PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda",
    },
    {
        "source_dir": "mg_panda_og_pnp",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-og-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-panda-og-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Panda-OG PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda-OG",
    },
    {
        "source_dir": "mg_jaco_pnp",
        "repo_id": "ajaysri/robocasa-x-sp900-jaco-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-jaco-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Jaco PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Jaco",
    },
    {
        "source_dir": "mg_panda_turn_on_sink",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-panda-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Panda Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Panda",
    },
    {
        "source_dir": "mg_panda_og_turn_on_sink",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-og-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-panda-og-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Panda-OG Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Panda-OG",
    },
    {
        "source_dir": "mg_jaco_turn_on_sink",
        "repo_id": "ajaysri/robocasa-x-sp900-jaco-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-jaco-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Jaco Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Jaco",
    },
    {
        "source_dir": "mg_panda_flip_mug",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-panda-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Panda Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Panda",
    },
    {
        "source_dir": "mg_panda_og_flip_mug",
        "repo_id": "ajaysri/robocasa-x-sp900-panda-og-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-panda-og-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Panda-OG Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Panda-OG",
    },
    {
        "source_dir": "mg_jaco_flip_mug",
        "repo_id": "ajaysri/robocasa-x-sp900-jaco-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-sp900-jaco-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X SP-900 Jaco Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Jaco",
    },
    {
        "source_dir": "panda_pnp",
        "repo_id": "ajaysri/robocasa-x-target-panda-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-panda-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Panda PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda",
    },
    {
        "source_dir": "panda_og_pnp",
        "repo_id": "ajaysri/robocasa-x-target-panda-og-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-panda-og-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Panda-OG PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda-OG",
    },
    {
        "source_dir": "jaco_pnp",
        "repo_id": "ajaysri/robocasa-x-target-jaco-pnp-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-jaco-pnp-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Jaco PnP VQ Action Tokenizer",
        "task": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Jaco",
    },
    {
        "source_dir": "panda_turn_on_sink",
        "repo_id": "ajaysri/robocasa-x-target-panda-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-panda-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Panda Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Panda",
    },
    {
        "source_dir": "panda_og_turn_on_sink",
        "repo_id": "ajaysri/robocasa-x-target-panda-og-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-panda-og-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Panda-OG Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Panda-OG",
    },
    {
        "source_dir": "jaco_turn_on_sink",
        "repo_id": "ajaysri/robocasa-x-target-jaco-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-jaco-turn-on-sink-faucet-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Jaco Turn On Sink Faucet VQ Action Tokenizer",
        "task": "Turn On Sink Faucet",
        "robots": "Jaco",
    },
    {
        "source_dir": "panda_flip_mug",
        "repo_id": "ajaysri/robocasa-x-target-panda-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-panda-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Panda Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Panda",
    },
    {
        "source_dir": "panda_og_flip_mug",
        "repo_id": "ajaysri/robocasa-x-target-panda-og-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-panda-og-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Panda-OG Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Panda-OG",
    },
    {
        "source_dir": "jaco_flip_mug",
        "repo_id": "ajaysri/robocasa-x-target-jaco-flip-mug-upright-vq-extra-action-tokenizer",
        "tokenizer_id": "robocasa-x-target-jaco-flip-mug-upright-vq-extra-action-tokenizer",
        "title": "RoboCasa-X Target Jaco Flip Mug Upright VQ Action Tokenizer",
        "task": "Flip Mug Upright",
        "robots": "Jaco",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("vq"),
        help="Root directory containing local VQ tokenizer directories.",
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=Path("/iliad2/u/ajaysri/robocasa_x_vq_stage"),
        help="Temporary staging root for README/.gitattributes augmented uploads.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of source tokenizer directories to upload.",
    )
    parser.add_argument(
        "--delete-first",
        action="store_true",
        help="Delete and recreate each target Hugging Face repo before uploading.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private.",
    )
    parser.add_argument(
        "--keep-staged",
        action="store_true",
        help="Keep staged directories after upload.",
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=8,
        help="Number of workers for upload_large_folder.",
    )
    return parser.parse_args()


def filtered_releases(only: List[str]) -> List[Dict[str, str]]:
    if not only:
        return ROBOCASA_X_VQ_TOKENIZER_RELEASES
    wanted = set(only)
    return [release for release in ROBOCASA_X_VQ_TOKENIZER_RELEASES if release["source_dir"] in wanted]


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _hardlink_or_copy(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def released_local_vq_dir(tokenizer_id: str) -> str:
    return tokenizer_id.replace("-vq-extra-action-tokenizer", "").replace("-", "_")


def build_readme(release: Dict[str, str]) -> str:
    return "\n".join(
        [
            "---",
            f"pretty_name: {release['title']}",
            "tags:",
            "  - robotics",
            "  - robocasa-x",
            "  - action-tokenizer",
            "---",
            "",
            f"# {release['title']}",
            "",
            f"- CLI tokenizer id: `{release['tokenizer_id']}`",
            f"- Task: `{release['task']}`",
            f"- Robots: `{release['robots']}`",
            f"- Expected local directory for current code: `vq/{released_local_vq_dir(release['tokenizer_id'])}`",
            "",
        ]
    )


def stage_release(release: Dict[str, str], source_root: Path, stage_root: Path) -> Path:
    source_dir = (source_root / release["source_dir"]).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing VQ source directory: {source_dir}")

    staged_dir = (stage_root / release["repo_id"].split("/", 1)[1]).resolve()
    if staged_dir.exists():
        shutil.rmtree(staged_dir)
    staged_dir.mkdir(parents=True, exist_ok=True)

    for file_path in _iter_files(source_dir):
        relative_path = file_path.relative_to(source_dir)
        _hardlink_or_copy(file_path, staged_dir / relative_path)

    (staged_dir / "README.md").write_text(build_readme(release))
    (staged_dir / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n")
    return staged_dir


def refresh_repo(api: HfApi, repo_id: str, private: bool, delete_first: bool) -> None:
    if delete_first:
        api.delete_repo(repo_id=repo_id, repo_type="model", missing_ok=True)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)


def upload_release(api: HfApi, release: Dict[str, str], staged_dir: Path, num_workers: int) -> None:
    api.upload_large_folder(
        repo_id=release["repo_id"],
        repo_type="model",
        folder_path=staged_dir,
        num_workers=num_workers,
    )


def main() -> None:
    args = parse_args()
    releases = filtered_releases(args.only)
    if not releases:
        raise ValueError("No RoboCasa-X VQ tokenizers matched the requested --only filter.")

    api = HfApi()
    for idx, release in enumerate(releases, start=1):
        print(f"[{idx}/{len(releases)}] {release['source_dir']} -> {release['repo_id']}", flush=True)
        staged_dir = stage_release(release, args.source_root, args.stage_root)
        refresh_repo(api, release["repo_id"], private=args.private, delete_first=args.delete_first)
        upload_release(api, release, staged_dir, num_workers=args.upload_workers)
        if not args.keep_staged and staged_dir.exists():
            shutil.rmtree(staged_dir)

    print("Completed RoboCasa-X VQ tokenizer upload.", flush=True)


if __name__ == "__main__":
    main()
