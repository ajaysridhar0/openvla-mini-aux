"""Build a BARX-compatible RLDS/TFDS dataset from rendered RoboCasa HDF5 files."""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds
from language_motion import LanguageMotionConverter, deduplicate_adjacent

IMAGE_SHAPE = (180, 320, 3)
MAX_OBJECTS = 1


def aligned_action(action: np.ndarray, action_space: str) -> np.ndarray:
    """Apply the exact Panda/non-Panda action selection used for RoboCasa-X."""
    if action_space == "panda":
        selected = action[:7]
    elif action.shape == (11,):
        selected = np.concatenate([action[:6], action[-1:]])
    else:
        selected = np.concatenate([action[:6], action[-2:-1]])
    if selected.shape != (7,):
        raise ValueError(f"Expected a 7-D aligned action, got {selected.shape} from {action.shape}")
    return selected.astype(np.float32)


def load_episode(
    path: Path,
    demo_id: str,
    action_space: str,
    include_wrist: bool,
    source_root: Path,
) -> Tuple[str, Dict[str, Any]]:
    """Load one rendered HDF5 demo and add BARX auxiliary labels."""
    with h5py.File(path, "r") as dataset:
        demo = dataset["data"][demo_id]
        actions = demo["actions"][()]
        ee_states = demo["obs"]["ee_states"][()]
        gripper_states = demo["obs"]["gripper_states"][()]
        images = demo["obs"]["agentview_rgb"][()]
        wrist_images = demo["obs"]["eye_in_hand_rgb"][()] if include_wrist else None
        ee_points = demo["aux_info"]["eef_normalized_image_pts"][()]
        bbox_group = demo["aux_info"]["bboxes_2d"]
        object_names = sorted(bbox_group.keys())[:MAX_OBJECTS]
        boxes = [bbox_group[name][()].reshape(-1, 1, 4) for name in object_names]
        if boxes:
            object_boxes = np.concatenate(boxes, axis=1)
            object_boxes = np.pad(object_boxes, ((0, 0), (0, MAX_OBJECTS - len(object_names)), (0, 0)))
        else:
            object_boxes = np.zeros((len(actions), MAX_OBJECTS, 4), dtype=np.float32)
        ep_meta = json.loads(demo.attrs["ep_meta"])

    lengths = {len(actions), len(ee_states), len(gripper_states), len(images), len(ee_points), len(object_boxes)}
    if wrist_images is not None:
        lengths.add(len(wrist_images))
    if len(lengths) != 1:
        raise ValueError(f"Mismatched trajectory lengths in {path}:{demo_id}: {sorted(lengths)}")
    if not len(actions):
        raise ValueError(f"Empty trajectory in {path}:{demo_id}")

    steps: List[Dict[str, Any]] = []
    for index, action in enumerate(actions):
        observation = {
            "image": images[index],
            "state": np.concatenate([ee_states[index], gripper_states[index]]).astype(np.float32),
        }
        if include_wrist:
            observation["wrist_image"] = wrist_images[index]
        steps.append(
            {
                "observation": observation,
                "action": aligned_action(action, action_space),
                "discount": np.float32(1.0),
                "reward": np.float32(index == len(actions) - 1),
                "is_first": index == 0,
                "is_last": index == len(actions) - 1,
                "is_terminal": index == len(actions) - 1,
                "language_instruction": ep_meta["lang"],
                "ee_pose_2D": np.asarray(ee_points[index], dtype=np.float32),
                "obj_bboxes": np.asarray(object_boxes[index], dtype=np.float32),
                "obj_bbox_names": "|".join(object_names),
            }
        )

    motions = LanguageMotionConverter().convert(steps)
    motions[-1] = "stop"
    if len(steps) > 1:
        steps[-1]["action"] = steps[-2]["action"].copy()
    steps[-1]["action"][:-1] = 0
    for index, step in enumerate(steps):
        step["language_motions"] = "|".join(deduplicate_adjacent(motions[: index + 1]))
        step["language_motions_future"] = "|".join(deduplicate_adjacent(motions[index + 1 :]))

    try:
        source_id = str(path.resolve().relative_to(source_root.resolve()))
    except ValueError:
        source_id = path.name
    key = f"{source_id}:{demo_id}"
    return key, {"steps": steps, "episode_metadata": {"file_path": source_id}}


class BarxRobocasa(tfds.core.GeneratorBasedBuilder):
    """Parameterized builder for the common BARX RoboCasa-X schema."""

    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES: ClassVar = {
        "1.0.0": "Original single-camera schema.",
        "1.1.0": "Optional wrist camera support.",
    }

    def __init__(self, *, source_files: List[Path], action_space: str, include_wrist: bool, source_root: Path, **kwargs):
        self.source_files = source_files
        self.action_space = action_space
        self.include_wrist = include_wrist
        self.source_root = source_root
        super().__init__(**kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        observation = {
            "image": tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="jpeg"),
            "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
        }
        if self.include_wrist:
            observation["wrist_image"] = tfds.features.Image(shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="jpeg")
        return tfds.core.DatasetInfo(
            builder=self,
            description="BARX-compatible RoboCasa-X episodes with behavior-aligned auxiliary labels.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(observation),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                            "language_motions": tfds.features.Text(),
                            "language_motions_future": tfds.features.Text(),
                            "ee_pose_2D": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                            "obj_bboxes": tfds.features.Tensor(shape=(MAX_OBJECTS, 4), dtype=np.float32),
                            "obj_bbox_names": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict({"file_path": tfds.features.Text()}),
                }
            ),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        return {"train": self._generate_examples(self.source_files)}

    def _generate_examples(self, paths: List[Path]) -> Iterator[Tuple[str, Dict[str, Any]]]:
        for path in paths:
            with h5py.File(path, "r") as dataset:
                demo_ids = sorted(dataset["data"], key=lambda name: int(name.rsplit("_", maxsplit=1)[-1]))
            for demo_id in demo_ids:
                yield load_episode(path, demo_id, self.action_space, self.include_wrist, self.source_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True, help="Glob for rendered demo_gentex_im320*.hdf5 files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="TFDS data directory.")
    parser.add_argument("--source-root", type=Path, help="Root stripped from non-sensitive episode source IDs.")
    parser.add_argument("--action-space", choices=("panda", "non-panda"), required=True)
    parser.add_argument("--no-wrist", action="store_false", dest="include_wrist", default=True)
    parser.add_argument("--max-files", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_files = [Path(path) for path in sorted(glob.glob(args.input_glob, recursive=True))]
    if args.max_files is not None:
        source_files = source_files[: args.max_files]
    if not source_files:
        raise FileNotFoundError(f"No files matched {args.input_glob!r}")
    source_root = args.source_root or Path(os.path.commonpath([str(path.parent) for path in source_files]))
    builder = BarxRobocasa(
        source_files=source_files,
        action_space=args.action_space,
        include_wrist=args.include_wrist,
        source_root=source_root,
        data_dir=str(args.output_dir),
    )
    builder.download_and_prepare()
    print(f"Wrote {builder.info.splits['train'].num_examples} episodes to {builder.data_path}")


if __name__ == "__main__":
    main()
