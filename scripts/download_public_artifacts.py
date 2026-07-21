#!/usr/bin/env python3
"""Download the pinned public artifacts for the BARX XP-900 PnP walkthrough."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MANIFEST_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "public_artifacts.json"
)


@dataclass(frozen=True)
class DownloadSpec:
    name: str
    repo_id: str
    repo_type: str
    revision: str
    local_dir: Optional[Path]
    allow_patterns: tuple[str, ...]
    expected_files: dict[str, dict]
    cache_only: bool = False


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _artifact_directories(args: argparse.Namespace) -> dict[str, Path]:
    if args.artifact_root is not None:
        root = args.artifact_root
        defaults = {
            "data_root": root / "data",
            "base_vlm_dir": root / "base-vlm",
            "vq_root": root / "vq",
            "run_root": root / "runs",
            "hf_home": root / "hf",
        }
    else:
        if args.data_root is None or args.base_vlm_dir is None:
            raise SystemExit(
                "Pass --artifact-root, or pass both --data-root and --base-vlm-dir."
            )
        sibling_root = args.base_vlm_dir.parent
        defaults = {
            "data_root": args.data_root,
            "base_vlm_dir": args.base_vlm_dir,
            "vq_root": sibling_root / "vq",
            "run_root": sibling_root / "runs",
            "hf_home": sibling_root / "hf",
        }

    return {
        name: (getattr(args, name) or default).expanduser().resolve()
        for name, default in defaults.items()
    }


def downloads(args: argparse.Namespace) -> tuple[list[DownloadSpec], Path]:
    manifest = load_manifest()
    directories = _artifact_directories(args)
    destination_overrides = {
        "base-vlm": directories["base_vlm_dir"],
        "data/mg_pnp_lite": directories["data_root"] / "mg_pnp_lite",
        "vq/mg_pnp_lite": directories["vq_root"] / "mg_pnp_lite",
        "runs/xp900-pnp-joint-reps": directories["run_root"] / "xp900-pnp-joint-reps",
    }

    specs = []
    for item in manifest["artifacts"]:
        if item.get("optional", False) and not args.include_pretrain_checkpoint:
            continue
        specs.append(
            DownloadSpec(
                name=item["name"],
                repo_id=item["repo_id"],
                repo_type=item["repo_type"],
                revision=item["revision"],
                local_dir=destination_overrides[item["destination"]],
                allow_patterns=tuple(item.get("allow_patterns", ())),
                expected_files=item.get("expected_files", {}),
            )
        )

    if not args.skip_runtime_cache:
        specs.extend(
            DownloadSpec(
                name=item["name"],
                repo_id=item["repo_id"],
                repo_type=item["repo_type"],
                revision=item["revision"],
                local_dir=None,
                allow_patterns=tuple(item.get("allow_patterns", ())),
                expected_files={},
                cache_only=True,
            )
            for item in manifest["runtime_dependencies"]
        )
    return specs, directories["hf_home"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="external root for data, checkpoints, VQ weights, runs, and the Hugging Face cache",
    )
    parser.add_argument("--data-root", type=Path, help="override the RLDS data root")
    parser.add_argument(
        "--base-vlm-dir", type=Path, help="override the base VLM directory"
    )
    parser.add_argument("--vq-root", type=Path, help="override the VQ tokenizer root")
    parser.add_argument(
        "--run-root", type=Path, help="override the source-prior run root"
    )
    parser.add_argument(
        "--hf-home", type=Path, help="override the Hugging Face cache root"
    )
    parser.add_argument("--include-pretrain-checkpoint", action="store_true")
    parser.add_argument(
        "--skip-runtime-cache",
        action="store_true",
        help="do not prefetch the pinned DINOv2, SigLIP, and Qwen runtime dependencies",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(spec: DownloadSpec) -> None:
    if spec.local_dir is None:
        return
    for relative_path, expected in spec.expected_files.items():
        path = spec.local_dir / relative_path
        if not path.is_file():
            raise RuntimeError(f"missing downloaded file: {path}")
        if path.stat().st_size != expected["size_bytes"]:
            raise RuntimeError(f"size mismatch for {path}")
        if _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path}")


def main() -> None:
    args = build_parser().parse_args()
    specs, hf_home = downloads(args)
    print(f"HF_HOME {hf_home}")

    for spec in specs:
        destination = hf_home / "hub" if spec.cache_only else spec.local_dir
        print(f"{spec.repo_type:7} {spec.repo_id}@{spec.revision} -> {destination}")
        if args.dry_run:
            continue

        from huggingface_hub import snapshot_download

        kwargs = {
            "repo_id": spec.repo_id,
            "repo_type": spec.repo_type,
            "revision": spec.revision,
            "allow_patterns": list(spec.allow_patterns) or None,
            "token": False,
        }
        if spec.cache_only:
            kwargs["cache_dir"] = hf_home / "hub"
        else:
            kwargs["local_dir"] = spec.local_dir
        snapshot_download(**kwargs)
        verify_files(spec)


if __name__ == "__main__":
    main()
