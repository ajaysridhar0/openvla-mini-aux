#!/usr/bin/env python3
"""
Create a repo-local `vq/` overlay where RoboCasa-X VQ directory names are canonical.

The shared BARX VQ tree still uses legacy directory names like `mg_pnp` and `panda_pnp`.
This script creates a writable local `vq/` directory in the repo root with:

- `vq/robocasa_x_*` symlinks pointing at the shared legacy directories
- legacy `vq/mg_*` / `vq/panda_*` / `vq/mg_panda_*` symlinks pointing back to the
  corresponding RoboCasa-X names for backward compatibility

Any unrelated entries from the shared VQ tree are also symlinked through unchanged.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict


ROBOCASA_X_VQ_DIR_MAPPING: Dict[str, str] = {
    "mg_pnp": "robocasa_x_xp3k_pnp",
    "mg_pnp_lite": "robocasa_x_xp900_pnp",
    "mg_turn_on_sink": "robocasa_x_xp3k_turn_on_sink_faucet",
    "mg_turn_on_sink_lite": "robocasa_x_xp900_turn_on_sink_faucet",
    "mg_flip_mug": "robocasa_x_xp3k_flip_mug_upright",
    "mg_flip_mug_lite": "robocasa_x_xp900_flip_mug_upright",
    "mg_panda_pnp": "robocasa_x_sp900_panda_pnp",
    "mg_panda_og_pnp": "robocasa_x_sp900_panda_og_pnp",
    "mg_jaco_pnp": "robocasa_x_sp900_jaco_pnp",
    "mg_panda_turn_on_sink": "robocasa_x_sp900_panda_turn_on_sink_faucet",
    "mg_panda_og_turn_on_sink": "robocasa_x_sp900_panda_og_turn_on_sink_faucet",
    "mg_jaco_turn_on_sink": "robocasa_x_sp900_jaco_turn_on_sink_faucet",
    "mg_panda_flip_mug": "robocasa_x_sp900_panda_flip_mug_upright",
    "mg_panda_og_flip_mug": "robocasa_x_sp900_panda_og_flip_mug_upright",
    "mg_jaco_flip_mug": "robocasa_x_sp900_jaco_flip_mug_upright",
    "panda_pnp": "robocasa_x_target_panda_pnp",
    "panda_og_pnp": "robocasa_x_target_panda_og_pnp",
    "jaco_pnp": "robocasa_x_target_jaco_pnp",
    "panda_turn_on_sink": "robocasa_x_target_panda_turn_on_sink_faucet",
    "panda_og_turn_on_sink": "robocasa_x_target_panda_og_turn_on_sink_faucet",
    "jaco_turn_on_sink": "robocasa_x_target_jaco_turn_on_sink_faucet",
    "panda_flip_mug": "robocasa_x_target_panda_flip_mug_upright",
    "panda_og_flip_mug": "robocasa_x_target_panda_og_flip_mug_upright",
    "jaco_flip_mug": "robocasa_x_target_jaco_flip_mug_upright",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path("vq"),
        help="Repo-local overlay directory to create. Defaults to `vq` in the current working tree.",
    )
    parser.add_argument(
        "--shared-vq-root",
        type=Path,
        default=None,
        help="Path to the shared legacy VQ root. If omitted and `overlay-dir` is currently a symlink, uses its target.",
    )
    parser.add_argument(
        "--backup-link-file",
        type=Path,
        default=Path(".vq_shared_symlink_backup"),
        help="Where to store the original `vq` symlink target if `overlay-dir` starts as a symlink.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing non-symlink overlay directory if it already exists.",
    )
    return parser.parse_args()


def infer_shared_root(overlay_dir: Path, explicit_shared_root: Path | None) -> Path:
    if explicit_shared_root is not None:
        return explicit_shared_root.resolve()
    if overlay_dir.is_symlink():
        return overlay_dir.resolve()
    raise ValueError("Pass --shared-vq-root or run this while `vq` is still a symlink to the shared legacy tree.")


def prepare_overlay_dir(overlay_dir: Path, backup_link_file: Path, force: bool) -> None:
    if overlay_dir.is_symlink():
        backup_link_file.write_text(os.readlink(overlay_dir) + "\n")
        overlay_dir.unlink()
    elif overlay_dir.exists():
        if not force:
            raise ValueError(f"{overlay_dir} already exists and is not a symlink. Re-run with --force to replace it.")
        shutil.rmtree(overlay_dir)

    overlay_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    overlay_dir = args.overlay_dir.resolve()
    shared_vq_root = infer_shared_root(args.overlay_dir, args.shared_vq_root)
    prepare_overlay_dir(args.overlay_dir, args.backup_link_file, args.force)

    for entry in sorted(shared_vq_root.iterdir()):
        if entry.name in ROBOCASA_X_VQ_DIR_MAPPING:
            continue
        os.symlink(str(entry.resolve()), overlay_dir / entry.name)

    for legacy_name, robocasa_name in ROBOCASA_X_VQ_DIR_MAPPING.items():
        shared_entry = (shared_vq_root / legacy_name).resolve()
        os.symlink(str(shared_entry), overlay_dir / robocasa_name)
        os.symlink(robocasa_name, overlay_dir / legacy_name)

    print(f"Created RoboCasa-X VQ overlay at {overlay_dir}")
    print(f"Shared legacy VQ root: {shared_vq_root}")
    print(f"Canonical RoboCasa-X directories: {len(ROBOCASA_X_VQ_DIR_MAPPING)}")


if __name__ == "__main__":
    main()
