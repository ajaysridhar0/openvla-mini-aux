"""Portable storage for deterministic BARX evaluation conditions."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np


FORMAT_VERSION = 2
PACKAGE_MARKERS = ("robocasa", "robosuite")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for hashing and comparison."""

    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def portable_ep_meta(value: Any) -> Any:
    """Replace installation-specific package prefixes in episode metadata."""

    if isinstance(value, dict):
        return {key: portable_ep_meta(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_ep_meta(item) for item in value]
    if isinstance(value, tuple):
        return [portable_ep_meta(item) for item in value]
    if isinstance(value, str):
        for package in PACKAGE_MARKERS:
            marker = f"/{package}/"
            if marker in value:
                suffix = value.split(marker, 1)[1]
                return f"<{package.upper()}>/{suffix}"
    return value


def materialize_ep_meta(value: Any, package_roots: dict[str, Path]) -> Any:
    """Resolve portable package paths for the current installation."""

    if isinstance(value, dict):
        return {
            key: materialize_ep_meta(item, package_roots)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [materialize_ep_meta(item, package_roots) for item in value]
    if isinstance(value, str):
        for package in PACKAGE_MARKERS:
            prefix = f"<{package.upper()}>/"
            if value.startswith(prefix):
                return str(package_roots[package] / value.removeprefix(prefix))
    return value


def stable_replay_metadata(ep_meta: dict[str, Any]) -> dict[str, Any]:
    """Return metadata fields RoboCasa does not mutate during scene loading."""

    portable = portable_ep_meta(ep_meta)
    return {key: value for key, value in portable.items() if key != "object_cfgs"}


def portable_model_xml(xml: str) -> str:
    """Replace package installation prefixes in MuJoCo asset paths."""

    for package in PACKAGE_MARKERS:
        xml = re.sub(
            rf'file="[^"]*/{package}/([^"]+)"',
            rf'file="<{package.upper()}>/\1"',
            xml,
        )
    return xml


def materialize_model_xml(xml: str, package_roots: dict[str, Path]) -> str:
    """Resolve portable MuJoCo asset paths for the current installation."""

    for package in PACKAGE_MARKERS:
        xml = xml.replace(
            f'file="<{package.upper()}>/',
            f'file="{package_roots[package]}/',
        )
    return xml


def state_sha256(state: np.ndarray) -> str:
    """Hash a simulator state with an explicit portable dtype and shape."""

    state = np.ascontiguousarray(state, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(np.asarray(state.shape, dtype="<i8").tobytes())
    digest.update(state.tobytes())
    return digest.hexdigest()


def model_sha256(xml: str) -> str:
    """Hash model XML while ignoring machine-specific absolute path prefixes."""

    portable = portable_model_xml(xml)
    return hashlib.sha256(portable.encode("utf-8")).hexdigest()


def make_entry(
    *,
    episode: int,
    seed: int,
    ep_meta: dict[str, Any],
    model_xml: str,
    policy_start_state: np.ndarray,
) -> dict[str, Any]:
    """Build the JSON portion of one frozen evaluation condition."""

    ep_meta = portable_ep_meta(ep_meta)
    state_hash = state_sha256(policy_start_state)
    metadata_hash = hashlib.sha256(canonical_json(ep_meta).encode("utf-8")).hexdigest()
    condition_id = hashlib.sha256(
        f"{episode}:{seed}:{metadata_hash}:{state_hash}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "condition_id": condition_id,
        "episode": episode,
        "seed": seed,
        "layout_id": ep_meta["layout_id"],
        "style_id": ep_meta["style_id"],
        "instruction": ep_meta.get("lang", ""),
        "metadata_sha256": metadata_hash,
        "model_sha256": model_sha256(model_xml),
        "state_sha256": state_hash,
        "ep_meta": ep_meta,
    }


def bundle_paths(root: Path, task: str, embodiment: str) -> tuple[Path, Path]:
    directory = root / task
    return directory / f"{embodiment}.json", directory / f"{embodiment}.npz"


def write_bundle(
    root: Path,
    *,
    task: str,
    embodiment: str,
    entries: list[dict[str, Any]],
    states: list[np.ndarray],
    model_xmls: list[str],
    provenance: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically write one task/embodiment condition bundle."""

    if len(entries) != len(states) or len(entries) != len(model_xmls):
        raise ValueError("Condition metadata, state, and model counts differ")
    if not entries:
        raise ValueError("Cannot write an empty condition bundle")

    state_matrix = np.stack([np.asarray(state, dtype=np.float64) for state in states])
    portable_xmls = [portable_model_xml(xml) for xml in model_xmls]
    for entry, state, xml in zip(entries, state_matrix, portable_xmls):
        if entry["state_sha256"] != state_sha256(state):
            raise ValueError(f"State hash mismatch for episode {entry['episode']}")
        if entry["model_sha256"] != model_sha256(xml):
            raise ValueError(f"Model hash mismatch for episode {entry['episode']}")

    json_path, state_path = bundle_paths(root, task, embodiment)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = json_path.with_suffix(".json.staging")
    state_tmp = state_path.with_suffix(".npz.staging")

    payload = {
        "format_version": FORMAT_VERSION,
        "task": task,
        "embodiment": embodiment,
        "episode_count": len(entries),
        "state_file": state_path.name,
        "provenance": provenance,
        "episodes": entries,
    }
    json_tmp.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    with state_tmp.open("wb") as output:
        np.savez_compressed(
            output,
            policy_start_states=state_matrix,
            model_xmls=np.asarray(portable_xmls, dtype=np.str_),
        )

    os.replace(state_tmp, state_path)
    os.replace(json_tmp, json_path)
    return json_path, state_path


def load_bundle(
    root: Path,
    *,
    task: str,
    embodiment: str,
) -> tuple[dict[str, Any], np.ndarray, list[str]]:
    """Load and fully validate one frozen condition bundle."""

    json_path, state_path = bundle_paths(root, task, embodiment)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported condition format in {json_path}")
    if (payload.get("task"), payload.get("embodiment")) != (task, embodiment):
        raise ValueError(f"Condition identity mismatch in {json_path}")
    if payload.get("state_file") != state_path.name:
        raise ValueError(f"Condition state filename mismatch in {json_path}")

    with np.load(state_path, allow_pickle=False) as archive:
        states = np.asarray(archive["policy_start_states"], dtype=np.float64)
        model_xmls = archive["model_xmls"].tolist()
    entries = payload["episodes"]
    if (
        len(entries) != len(states)
        or len(entries) != len(model_xmls)
        or len(entries) != payload["episode_count"]
    ):
        raise ValueError(f"Condition episode count mismatch in {json_path}")

    for expected_episode, (entry, state, xml) in enumerate(
        zip(entries, states, model_xmls)
    ):
        if entry["episode"] != expected_episode:
            raise ValueError(f"Non-contiguous episode index in {json_path}")
        if entry["state_sha256"] != state_sha256(state):
            raise ValueError(f"Corrupt state for episode {expected_episode} in {state_path}")
        if entry["model_sha256"] != model_sha256(xml):
            raise ValueError(f"Corrupt model for episode {expected_episode} in {state_path}")
        metadata_hash = hashlib.sha256(
            canonical_json(entry["ep_meta"]).encode("utf-8")
        ).hexdigest()
        if entry["metadata_sha256"] != metadata_hash:
            raise ValueError(f"Corrupt metadata for episode {expected_episode} in {json_path}")

    return payload, states, model_xmls
