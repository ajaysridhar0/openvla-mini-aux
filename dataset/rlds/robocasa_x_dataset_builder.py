"""Normalized RoboCasa-X HDF5 to RLDS converter used by BARX.

All released HDF5 files use the same 12-D action order. This builder extracts
the first seven policy-controlled values with
``barx.action_space.canonicalize_action``.

Set ``BARX_RAW_DATA_GLOB`` to one or more ``os.pathsep``-separated HDF5 glob
patterns before invoking TFDS. For example::

    BARX_RAW_DATA_GLOB='/data/mg/*Omron/PnP*/*/demo_gentex_im320_randcams.hdf5' \
      tfds build dataset/rlds/robocasa_x_dataset_builder.py
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds

from barx.action_space import canonicalize_action

try:
    from .conversion_utils import MultiThreadedDatasetBuilder
    from .language_motion_converter import LanguageMotionEpisodeConverter, dedup_lm
except ImportError:  # TFDS may load a builder file outside its package context.
    from conversion_utils import MultiThreadedDatasetBuilder
    from language_motion_converter import LanguageMotionEpisodeConverter, dedup_lm


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
MAX_NUM_OBJECTS_OF_INTEREST = 1
LANGUAGE_MOTION_CONVERTER = LanguageMotionEpisodeConverter()


def _parse_episode(episode_path: str, demo_id: str) -> Tuple[str, Any]:
    with h5py.File(episode_path, "r") as source:
        demo = source["data"][demo_id]
        actions = demo["actions"][()]
        states = demo["obs"]["ee_states"][()]
        gripper_states = demo["obs"]["gripper_states"][()]
        images = demo["obs"]["agentview_rgb"][()]
        ee_positions_2d = demo["aux_info"]["eef_normalized_image_pts"][()]

        bounding_boxes = demo["aux_info"]["bboxes_2d"]
        object_names = sorted(bounding_boxes.keys())[:MAX_NUM_OBJECTS_OF_INTEREST]
        object_box_arrays = [bounding_boxes[name][()].reshape(-1, 1, 4) for name in object_names]
        if object_box_arrays:
            object_boxes = np.concatenate(object_box_arrays, axis=1)
            object_boxes = np.pad(
                object_boxes,
                ((0, 0), (0, max(0, MAX_NUM_OBJECTS_OF_INTEREST - object_boxes.shape[1])), (0, 0)),
                "constant",
            )
        else:
            object_boxes = np.zeros((states.shape[0], MAX_NUM_OBJECTS_OF_INTEREST, 4))

        metadata = json.loads(demo.attrs["ep_meta"])
        instruction = metadata["lang"]

    episode = []
    for index, raw_action in enumerate(actions):
        action = canonicalize_action(raw_action)
        episode.append(
            {
                "observation": {
                    "image": images[index],
                    "state": np.asarray(
                        np.concatenate((states[index], gripper_states[index]), axis=-1),
                        dtype=np.float32,
                    ),
                },
                "action": np.asarray(action, dtype=np.float32),
                "discount": 1.0,
                "reward": float(index == len(actions) - 1),
                "is_first": index == 0,
                "is_last": index == len(actions) - 1,
                "is_terminal": index == len(actions) - 1,
                "language_instruction": instruction,
                # Stored field names remain unchanged for model compatibility.
                "ee_pose_2D": np.asarray(ee_positions_2d[index], dtype=np.float32),
                "obj_bboxes": np.asarray(object_boxes[index], dtype=np.float32),
                "obj_bbox_names": "|".join(object_names),
            }
        )

    language_motions = LANGUAGE_MOTION_CONVERTER.convert(episode)
    language_motions[-1] = "stop"
    episode[-1]["action"] = episode[-2]["action"].copy()
    episode[-1]["action"][:-1] = 0

    for index in range(len(actions)):
        previous = dedup_lm(language_motions[: index + 1])
        future = dedup_lm(language_motions[index + 1 :])
        episode[index]["language_motions"] = "|".join(previous)
        episode[index]["language_motions_future"] = "|".join(future)

    return f"{episode_path}_{demo_id}", {
        "steps": episode,
        "episode_metadata": {"file_path": episode_path},
    }


def _generate_examples(paths: list[str]) -> Iterator[Tuple[str, Any]]:
    for episode_path in paths:
        with h5py.File(episode_path, "r") as source:
            demo_ids = list(source["data"].keys())
        for demo_id in demo_ids:
            yield _parse_episode(episode_path, demo_id)


class RobocasaXDataset(MultiThreadedDatasetBuilder):
    """Unified RLDS builder for every BARX RoboCasa-X embodiment."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Unified BARX simulation dataset release."}
    BUILDER_CONFIGS = []
    DEFAULT_BUILDER_CONFIG_NAME = None
    MANUAL_DOWNLOAD_INSTRUCTIONS = None
    N_WORKERS = int(os.environ.get("BARX_CONVERTER_WORKERS", "32"))
    MAX_PATHS_IN_MEMORY = int(os.environ.get("BARX_CONVERTER_PATH_BATCH", "160"))
    PARSE_FCN = _generate_examples

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,),
                                        dtype=np.float32,
                                        doc="Robot end-effector state (6-D pose, 2-D gripper).",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,), dtype=np.float32, doc="Canonical BARX end-effector action."
                            ),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                            "language_motions": tfds.features.Text(),
                            "language_motions_future": tfds.features.Text(),
                            "ee_pose_2D": tfds.features.Tensor(shape=(2,), dtype=np.float32),
                            "obj_bboxes": tfds.features.Tensor(
                                shape=(MAX_NUM_OBJECTS_OF_INTEREST, 4), dtype=np.float32
                            ),
                            "obj_bbox_names": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {"file_path": tfds.features.Text(doc="Original HDF5 path.")}
                    ),
                }
            )
        )

    def _split_paths(self) -> dict[str, list[str]]:
        patterns = os.environ.get("BARX_RAW_DATA_GLOB", "")
        if not patterns:
            raise ValueError("Set BARX_RAW_DATA_GLOB to one or more HDF5 glob patterns.")

        files = []
        for pattern in patterns.split(os.pathsep):
            files.extend(glob.glob(pattern, recursive=True))
        files = sorted(set(files))

        valid_files = []
        for path in files:
            try:
                with h5py.File(path, "r") as source:
                    if "data" in source:
                        valid_files.append(path)
            except OSError:
                continue
        if not valid_files:
            raise FileNotFoundError(f"No valid HDF5 files matched: {patterns}")
        return {"train": valid_files}
