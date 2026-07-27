"""Portable storage for deterministic BARX evaluation conditions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from barx.benchmark import OBJECT_CATEGORIES, OBJECT_GROUP, OBJECT_INSTANCE_SPLIT

FORMAT_VERSION = 3
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
            key: materialize_ep_meta(item, package_roots) for key, item in value.items()
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


def validate_condition_semantics(entry: dict[str, Any], *, task: str) -> None:
    """Reject frozen conditions whose instruction, target set, or source fixture disagree."""

    metadata = entry["ep_meta"]
    object_cfgs = {cfg["name"]: cfg for cfg in metadata.get("object_cfgs", [])}
    if task == "turn_on_sink_faucet":
        if entry["instruction"] != "turn on the sink faucet":
            raise ValueError(
                f"Condition {entry['condition_id']} has the wrong faucet instruction"
            )
        return

    target = object_cfgs.get("obj")
    if target is None:
        raise ValueError(f"Condition {entry['condition_id']} has no target object")
    target_info = target.get("info", {})
    target_category = target_info.get("cat")
    if not target_category:
        raise ValueError(
            f"Condition {entry['condition_id']} has no target object category"
        )
    if target_info.get("split") != OBJECT_INSTANCE_SPLIT:
        raise ValueError(
            f"Condition {entry['condition_id']} target is not from object split "
            f"{OBJECT_INSTANCE_SPLIT}"
        )
    if task.startswith("pnp_") and target_category not in OBJECT_CATEGORIES:
        raise ValueError(
            f"Condition {entry['condition_id']} target category "
            f"{target_category!r} is not in {OBJECT_GROUP}"
        )

    fixture_refs = metadata.get("fixture_refs", {})
    if task == "pnp_counter_to_sink":
        expected_fixture = fixture_refs.get("counter")
        expected_instruction = (
            f"pick the {target_category} from the counter and place it in the sink"
        )
    elif task == "pnp_sink_to_counter":
        expected_fixture = fixture_refs.get("sink")
        container = object_cfgs.get("container")
        if container is None:
            raise ValueError(
                f"Condition {entry['condition_id']} has no target receptacle"
            )
        if container.get("placement", {}).get("fixture") != fixture_refs.get("counter"):
            raise ValueError(
                f"Condition {entry['condition_id']} receptacle is not on the counter"
            )
        container_category = container.get("info", {}).get("cat")
        expected_instruction = (
            f"pick the {target_category} from the sink and place it on the "
            f"{container_category} located on the counter"
        )
    elif task == "flip_mug_upright":
        expected_fixture = fixture_refs.get("counter")
        expected_instruction = "flip the mug on the counter upright"
        if target.get("obj_groups") != "mug" or target_category != "mug":
            raise ValueError(f"Condition {entry['condition_id']} does not target a mug")
    else:
        raise ValueError(f"Unknown BARX evaluation task: {task}")

    if target.get("placement", {}).get("fixture") != expected_fixture:
        raise ValueError(
            f"Condition {entry['condition_id']} target is not on the task source fixture"
        )
    if entry["instruction"] != expected_instruction:
        raise ValueError(
            f"Condition {entry['condition_id']} instruction disagrees with its target"
        )
    if task.startswith("pnp_"):
        containing_groups = target_info.get("groups_containing_sampled_obj", [])
        if (
            target.get("obj_groups") != OBJECT_GROUP
            or OBJECT_GROUP not in containing_groups
        ):
            raise ValueError(
                f"Condition {entry['condition_id']} target is not from "
                f"{OBJECT_GROUP}"
            )


def _xml_root_for_analysis(model_xml: str) -> ET.Element:
    for package in PACKAGE_MARKERS:
        model_xml = model_xml.replace(f"<{package.upper()}>", f"__{package.upper()}__")
    return ET.fromstring(model_xml)


def _numbers(value: str | None, size: int) -> np.ndarray:
    if value is None:
        return np.zeros(size, dtype=np.float64)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.size != size:
        raise ValueError(f"Expected {size} numbers, received {value!r}")
    return result


def _quaternion_matrix(value: str | None) -> np.ndarray:
    if value is None:
        return np.eye(3, dtype=np.float64)
    quaternion = _numbers(value, 4)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Camera transform has a zero quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Joint has a zero axis")
    x, y, z = axis / norm
    cosine = math.cos(angle)
    sine = math.sin(angle)
    complement = 1 - cosine
    return np.asarray(
        [
            [
                cosine + x * x * complement,
                x * y * complement - z * sine,
                x * z * complement + y * sine,
            ],
            [
                y * x * complement + z * sine,
                cosine + y * y * complement,
                y * z * complement - x * sine,
            ],
            [
                z * x * complement - y * sine,
                z * y * complement + x * sine,
                cosine + z * z * complement,
            ],
        ],
        dtype=np.float64,
    )


def _static_camera_pose(
    root: ET.Element,
    camera_name: str,
    joint_positions: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Frozen model XML has no worldbody")

    def locate(element: ET.Element) -> list[ET.Element] | None:
        if element.tag == "camera" and element.get("name") == camera_name:
            return [element]
        for child in element:
            if child.tag not in {"body", "camera"}:
                continue
            suffix = locate(child)
            if suffix is not None:
                return [element, *suffix]
        return None

    path = None
    for body in worldbody.findall("body"):
        path = locate(body)
        if path is not None:
            break
    if path is None:
        raise ValueError(f"Frozen model XML has no {camera_name} camera")

    position = np.zeros(3, dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    field_of_view = 45.0
    for element in path:
        if element.get("euler") is not None or element.get("axisangle") is not None:
            raise ValueError("Evaluation camera path must use quaternion rotations")
        position = position + rotation @ _numbers(element.get("pos"), 3)
        rotation = rotation @ _quaternion_matrix(element.get("quat"))
        for child in element:
            if child.tag not in {"joint", "freejoint"}:
                continue
            joint_name = child.get("name", "")
            if not joint_name or joint_name not in joint_positions:
                raise ValueError(f"Frozen state has no {joint_name or 'unnamed'} joint")
            joint_type = (
                "free" if child.tag == "freejoint" else child.get("type", "hinge")
            )
            joint_value = joint_positions[joint_name]
            if joint_type == "slide":
                position = position + rotation @ (
                    _numbers(child.get("axis"), 3) * float(joint_value[0])
                )
            elif joint_type == "hinge":
                joint_position = _numbers(child.get("pos"), 3)
                joint_rotation = _axis_angle_matrix(
                    _numbers(child.get("axis"), 3), float(joint_value[0])
                )
                position = (
                    position
                    + rotation @ joint_position
                    - rotation @ joint_rotation @ joint_position
                )
                rotation = rotation @ joint_rotation
            elif not np.allclose(joint_value, 0.0, atol=1e-10):
                raise ValueError(
                    f"Camera {camera_name} requires unsupported {joint_type} projection"
                )
        if element.tag == "camera":
            field_of_view = float(element.get("fovy", "45"))
    return position, rotation, field_of_view


def _target_position(
    root: ET.Element, state: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    joints: list[tuple[str, str]] = []

    def visit(body: ET.Element) -> None:
        for child in body:
            if child.tag in {"joint", "freejoint"}:
                joint_type = (
                    "free" if child.tag == "freejoint" else child.get("type", "hinge")
                )
                joints.append((child.get("name", ""), joint_type))
        # MuJoCo stores every joint on a body before joints on descendant
        # bodies, regardless of the interleaving of XML child elements.
        for child in body:
            if child.tag == "body":
                visit(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Frozen model XML has no worldbody")
    for body in worldbody.findall("body"):
        visit(body)

    qpos_size = sum(
        7 if joint_type == "free" else 4 if joint_type == "ball" else 1
        for _, joint_type in joints
    )
    qvel_size = sum(
        6 if joint_type == "free" else 3 if joint_type == "ball" else 1
        for _, joint_type in joints
    )
    if np.asarray(state).size != 1 + qpos_size + qvel_size:
        raise ValueError("Frozen state shape disagrees with model joint layout")

    qpos = np.asarray(state, dtype=np.float64)[1 : 1 + qpos_size]
    address = 0
    joint_positions = {}
    target_position = None
    for joint_name, joint_type in joints:
        size = 7 if joint_type == "free" else 4 if joint_type == "ball" else 1
        joint_positions[joint_name] = qpos[address : address + size].copy()
        if joint_name == "obj_joint0":
            if joint_type != "free":
                raise ValueError("Target object joint is not free")
            target_position = qpos[address : address + 3].copy()
        address += size
    if target_position is None:
        raise ValueError("Frozen model XML has no obj_joint0 target")
    return target_position, joint_positions


def project_target_center(
    entry: dict[str, Any],
    state: np.ndarray,
    model_xml: str,
    *,
    camera_name: str,
    image_width: int,
    image_height: int,
) -> dict[str, float | bool]:
    """Project the frozen target center into the policy camera without loading assets."""

    root = _xml_root_for_analysis(model_xml)
    target_position, joint_positions = _target_position(root, state)
    camera_position, camera_rotation, field_of_view = _static_camera_pose(
        root, camera_name, joint_positions
    )
    camera_point = camera_rotation.T @ (target_position - camera_position)
    depth = -float(camera_point[2])
    focal_length = 0.5 * image_height / math.tan(math.radians(field_of_view) / 2)
    if depth <= 0:
        pixel_x = pixel_y = math.nan
        in_frame = False
    else:
        pixel_x = image_width / 2 + focal_length * float(camera_point[0]) / depth
        pixel_y = image_height / 2 - focal_length * float(camera_point[1]) / depth
        in_frame = 0 <= pixel_x < image_width and 0 <= pixel_y < image_height
    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "depth": depth,
        "center_in_frame": in_frame,
    }


def target_visible_pixel_count(
    env: Any,
    *,
    camera_name: str,
    image_width: int,
    image_height: int,
) -> int:
    """Count policy-camera segmentation pixels belonging to the target object."""

    import mujoco

    target_body = env.env.obj_body_id["obj"]
    model = env.env.sim.model
    target_bodies = set()
    for body_id in range(model.nbody):
        ancestor = body_id
        while ancestor > 0:
            if ancestor == target_body:
                target_bodies.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    target_geoms = [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_bodyid[geom_id]) in target_bodies
    ]
    segmentation = env.env.sim.render(
        camera_name=camera_name,
        width=image_width,
        height=image_height,
        segmentation=True,
    )
    object_types = segmentation[..., 0]
    object_ids = segmentation[..., 1]
    mask = (object_types == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(
        object_ids, target_geoms
    )
    return int(np.count_nonzero(mask))


def validate_condition_bundle_for_evaluation(
    payload: dict[str, Any],
    states: list[np.ndarray],
    model_xmls: list[str],
    *,
    task: str,
    camera_name: str,
    image_width: int,
    image_height: int,
    count: int | None = None,
) -> None:
    """Require requested frozen targets to match the task and appear in-frame."""

    entries = payload["episodes"]
    count = len(entries) if count is None else count
    if count < 1 or count > len(entries):
        raise ValueError(
            f"Requested {count} conditions from a bundle containing {len(entries)}"
        )
    invalid: list[str] = []
    for entry, state, model_xml in zip(
        entries[:count], states[:count], model_xmls[:count]
    ):
        try:
            validate_condition_semantics(entry, task=task)
            if task != "turn_on_sink_faucet":
                projection = project_target_center(
                    entry,
                    state,
                    model_xml,
                    camera_name=camera_name,
                    image_width=image_width,
                    image_height=image_height,
                )
                if not projection["center_in_frame"]:
                    invalid.append(
                        f"seed {entry['seed']} target projects to "
                        f"({projection['pixel_x']:.1f}, {projection['pixel_y']:.1f})"
                    )
        except ValueError as error:
            invalid.append(f"seed {entry['seed']}: {error}")
    if invalid:
        preview = "; ".join(invalid[:5])
        remaining = len(invalid) - min(len(invalid), 5)
        suffix = f"; plus {remaining} more" if remaining else ""
        raise ValueError(
            f"Frozen evaluation bundle contains {len(invalid)} invalid requested "
            f"conditions: {preview}{suffix}. Regenerate with the visible-target "
            "condition protocol before evaluation."
        )


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
        f"{episode}:{seed}:{metadata_hash}:{state_hash}".encode()
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
        "state_size": int(np.asarray(policy_start_state).size),
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

    normalized_states = [
        np.asarray(state, dtype=np.float64).reshape(-1) for state in states
    ]
    state_offsets = np.zeros(len(normalized_states) + 1, dtype=np.int64)
    state_offsets[1:] = np.cumsum(
        [state.size for state in normalized_states], dtype=np.int64
    )
    state_values = np.concatenate(normalized_states)
    portable_xmls = [portable_model_xml(xml) for xml in model_xmls]
    for entry, state, xml in zip(entries, normalized_states, portable_xmls):
        if entry["state_size"] != state.size:
            raise ValueError(f"State size mismatch for episode {entry['episode']}")
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
            policy_start_state_values=state_values,
            policy_start_state_offsets=state_offsets,
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
) -> tuple[dict[str, Any], list[np.ndarray], list[str]]:
    """Load and fully validate one frozen condition bundle."""

    json_path, state_path = bundle_paths(root, task, embodiment)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported condition format in {json_path}")
    if (payload.get("task"), payload.get("embodiment")) != (task, embodiment):
        raise ValueError(f"Condition identity mismatch in {json_path}")
    if payload.get("state_file") != state_path.name:
        raise ValueError(f"Condition state filename mismatch in {json_path}")

    entries = payload["episodes"]
    with np.load(state_path, allow_pickle=False) as archive:
        state_values = np.asarray(
            archive["policy_start_state_values"], dtype=np.float64
        )
        state_offsets = np.asarray(
            archive["policy_start_state_offsets"], dtype=np.int64
        )
        model_xmls = archive["model_xmls"].tolist()
    if (
        state_values.ndim != 1
        or state_offsets.ndim != 1
        or len(state_offsets) != len(entries) + 1
        or state_offsets[0] != 0
        or np.any(np.diff(state_offsets) < 0)
        or state_offsets[-1] != len(state_values)
    ):
        raise ValueError(f"Corrupt state index in {state_path}")
    states = [
        state_values[state_offsets[index] : state_offsets[index + 1]].copy()
        for index in range(len(entries))
    ]
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
        if entry["state_size"] != state.size:
            raise ValueError(
                f"State size mismatch for episode {expected_episode} in {state_path}"
            )
        if entry["state_sha256"] != state_sha256(state):
            raise ValueError(
                f"Corrupt state for episode {expected_episode} in {state_path}"
            )
        if entry["model_sha256"] != model_sha256(xml):
            raise ValueError(
                f"Corrupt model for episode {expected_episode} in {state_path}"
            )
        metadata_hash = hashlib.sha256(
            canonical_json(entry["ep_meta"]).encode("utf-8")
        ).hexdigest()
        if entry["metadata_sha256"] != metadata_hash:
            raise ValueError(
                f"Corrupt metadata for episode {expected_episode} in {json_path}"
            )

    return payload, states, model_xmls


def restore_frozen_condition(
    env: Any,
    entry: dict[str, Any],
    state: np.ndarray,
    model_xml: str,
) -> tuple[Any, dict[str, Any]]:
    """Restore and verify one condition in an initialized RoboCasa environment."""

    import robocasa
    import robosuite

    package_roots = {
        "robocasa": Path(robocasa.__path__[0]),
        "robosuite": Path(robosuite.__path__[0]),
    }
    ep_meta = materialize_ep_meta(entry["ep_meta"], package_roots)
    env.env.set_ep_meta(ep_meta)
    env.reset(unset_ep_meta=False)

    model_xml = materialize_model_xml(model_xml, package_roots)
    actual_model_hash = model_sha256(model_xml)
    if actual_model_hash != entry["model_sha256"]:
        raise RuntimeError(
            f"Model mismatch for condition {entry['condition_id']}: "
            f"expected {entry['model_sha256']}, received {actual_model_hash}"
        )

    # Frozen XML has already passed through robosuite's XML processors and
    # must be loaded verbatim.
    xml_processors = env.env._xml_processors
    env.env._xml_processors = []
    try:
        env.env.reset_from_xml_string(model_xml)
    finally:
        env.env._xml_processors = xml_processors

    env.env.sim.set_state_from_flattened(state)
    env.env.sim.forward()
    restored_state = np.asarray(env.env.sim.get_state().flatten(), dtype=np.float64)
    if state_sha256(restored_state) != entry["state_sha256"]:
        raise RuntimeError(f"State mismatch for condition {entry['condition_id']}")

    restored_meta = env.env.get_ep_meta()
    if canonical_json(stable_replay_metadata(restored_meta)) != canonical_json(
        stable_replay_metadata(entry["ep_meta"])
    ):
        raise RuntimeError(f"Metadata mismatch for condition {entry['condition_id']}")
    return env.get_observation(), restored_meta
