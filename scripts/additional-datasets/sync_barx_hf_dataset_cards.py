#!/usr/bin/env python3
"""
Upload minimal README cards to the RoboCasa-X dataset repos on Hugging Face.
"""

import argparse
import tempfile
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

ROBOCASA_X_TASK_METADATA: Dict[str, Dict[str, str]] = {
    "pnp": {
        "title": "PnP Counter to Sink / PnP Sink to Counter",
        "robots": "Panda, Panda-OG, Jaco",
        "xp3k_demos": "6000",
        "xp900_demos": "1800",
        "sp900_demos": "1800",
        "target_demos": "100",
    },
    "turn_on_sink_faucet": {
        "title": "Turn On Sink Faucet",
        "robots": "Panda, Panda-OG, Jaco",
        "xp3k_demos": "3000",
        "xp900_demos": "900",
        "sp900_demos": "900",
        "target_demos": "50",
    },
    "flip_mug_upright": {
        "title": "Flip Mug Upright",
        "robots": "Panda, Panda-OG, Jaco",
        "xp3k_demos": "3000",
        "xp900_demos": "900",
        "sp900_demos": "900",
        "target_demos": "50",
    },
}

ROBOCASA_X_ROBOT_LABELS: Dict[str, str] = {
    "panda": "Panda",
    "panda_og": "Panda-OG",
    "jaco": "Jaco",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of target TFDS names to update, e.g. robocasa_x_xp3k_pnp.",
    )
    return parser.parse_args()


def filtered_releases(only: List[str]) -> List[Dict[str, str]]:
    if not only:
        return ROBOCASA_X_HF_DATASET_RELEASES
    wanted = set(only)
    return [release for release in ROBOCASA_X_HF_DATASET_RELEASES if release["target_name"] in wanted]


def infer_robocasa_x_release_metadata(target_name: str) -> Dict[str, str]:
    if target_name.startswith("robocasa_x_xp3k_"):
        task_key = target_name[len("robocasa_x_xp3k_") :]
        return {
            "title_prefix": "RoboCasa-X XP-3K",
            "split_label": "XP-3K",
            "task_key": task_key,
            "robots": ROBOCASA_X_TASK_METADATA[task_key]["robots"],
            "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["xp3k_demos"],
        }

    if target_name.startswith("robocasa_x_xp900_"):
        task_key = target_name[len("robocasa_x_xp900_") :]
        return {
            "title_prefix": "RoboCasa-X XP-900",
            "split_label": "XP-900",
            "task_key": task_key,
            "robots": ROBOCASA_X_TASK_METADATA[task_key]["robots"],
            "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["xp900_demos"],
        }

    if target_name.startswith("robocasa_x_sp900_"):
        remainder = target_name[len("robocasa_x_sp900_") :]
        for robot_key in ("panda_og", "panda", "jaco"):
            robot_prefix = f"{robot_key}_"
            if remainder.startswith(robot_prefix):
                task_key = remainder[len(robot_prefix) :]
                return {
                    "title_prefix": f"RoboCasa-X SP-900 {ROBOCASA_X_ROBOT_LABELS[robot_key]}",
                    "split_label": "SP-900",
                    "task_key": task_key,
                    "robots": ROBOCASA_X_ROBOT_LABELS[robot_key],
                    "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["sp900_demos"],
                }

    if target_name.startswith("robocasa_x_target_"):
        remainder = target_name[len("robocasa_x_target_") :]
        for robot_key in ("panda_og", "panda", "jaco"):
            robot_prefix = f"{robot_key}_"
            if remainder.startswith(robot_prefix):
                task_key = remainder[len(robot_prefix) :]
                return {
                    "title_prefix": f"RoboCasa-X Target {ROBOCASA_X_ROBOT_LABELS[robot_key]}",
                    "split_label": "Target",
                    "task_key": task_key,
                    "robots": ROBOCASA_X_ROBOT_LABELS[robot_key],
                    "num_demos": ROBOCASA_X_TASK_METADATA[task_key]["target_demos"],
                }

    return {
        "title_prefix": "RoboCasa-X Dataset",
        "split_label": "RoboCasa-X",
        "task_key": "",
        "robots": "",
        "num_demos": "",
    }


def build_dataset_card(target_name: str) -> str:
    metadata = infer_robocasa_x_release_metadata(target_name)
    task_metadata = ROBOCASA_X_TASK_METADATA.get(metadata["task_key"], {})
    task_title = task_metadata.get("title", target_name.replace("_", " "))
    title = f"{metadata['title_prefix']} {task_title}".strip()

    return "\n".join(
        [
            "---",
            f"pretty_name: {title}",
            "tags:",
            "  - robotics",
            "  - robocasa-x",
            "  - tensorflow-datasets",
            "---",
            "",
            f"# {title}",
            "",
            f"- Task: `{task_title}`",
            f"- TFDS builder id after download: `{target_name}`",
            f"- Robots in dataset: `{metadata['robots']}`",
            f"- Number of demos: `{metadata['num_demos']}`",
            "",
        ]
    )


def upload_dataset_card(api: HfApi, repo_id: str, target_name: str) -> None:
    content = build_dataset_card(target_name)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)

    try:
        api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            path_or_fileobj=str(temp_path),
            path_in_repo="README.md",
            commit_message=f"Update README for {target_name}",
        )
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    releases = filtered_releases(args.only)
    api = HfApi()

    if not releases:
        raise ValueError("No BARX dataset releases matched the requested --only filter.")

    for release in releases:
        print(f"Updating {release['repo_id']} README", flush=True)
        upload_dataset_card(api, release["repo_id"], release["target_name"])

    print("Completed BARX HF dataset card sync.", flush=True)


if __name__ == "__main__":
    main()
