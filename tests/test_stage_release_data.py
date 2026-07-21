import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.stage_release_data import (
    BODY_PART_ORDER,
    LAYOUT_ATTRIBUTE,
    NORMALIZED_LAYOUT,
    normalize_hdf5,
    stage_file,
    verify_hdf5,
)


class StageReleaseDataTest(unittest.TestCase):
    def make_file(self, path: Path, actions: np.ndarray, *, marked: bool = False) -> None:
        env_args = {
            "env_name": "PnPCounterToSink",
            "env_kwargs": {
                "controller_configs": {
                    "type": "HYBRID_MOBILE_BASE",
                    "composite_controller_specific_configs": {},
                }
            },
        }
        with h5py.File(path, "w") as output:
            data = output.create_group("data")
            data.attrs["env_args"] = json.dumps(env_args)
            if marked:
                data.attrs[LAYOUT_ATTRIBUTE] = NORMALIZED_LAYOUT
            demo = data.create_group("demo_0")
            demo.create_dataset("actions", data=actions)
            demo.attrs["ep_meta"] = json.dumps(
                {"object_cfgs": [{"info": {"mjcf_path": "/Users/private/robocasa/models/assets/mug.xml"}}]}
            )
            demo.attrs["model_file"] = "<mujoco file='/home/private/arm.xml'/>"

    def test_unmarked_non_panda_actions_are_permuted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jaco.hdf5"
            original = np.arange(24, dtype=np.float32).reshape(2, 12)
            self.make_file(path, original)
            self.assertEqual(normalize_hdf5(path, "JacoOmron"), 1)
            with h5py.File(path) as staged:
                np.testing.assert_array_equal(
                    staged["data/demo_0/actions"][...],
                    original[..., [0, 1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 11]],
                )
                env_args = json.loads(staged["data"].attrs["env_args"])
                ordering = env_args["env_kwargs"]["controller_configs"][
                    "composite_controller_specific_configs"
                ]["body_part_ordering"]
                self.assertEqual(ordering, BODY_PART_ORDER)

    def test_marked_source_is_not_permuted_again(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.hdf5", root / "release.hdf5"
            normalized = np.arange(12, dtype=np.float32).reshape(1, 12)
            self.make_file(source, normalized, marked=True)
            stage_file(source, destination, "JacoOmron", expected_demos=1)
            with h5py.File(source) as archival, h5py.File(destination) as staged:
                np.testing.assert_array_equal(staged["data/demo_0/actions"][...], normalized)
                np.testing.assert_array_equal(archival["data/demo_0/actions"][...], normalized)
            verify_hdf5(destination, 1)


if __name__ == "__main__":
    unittest.main()
