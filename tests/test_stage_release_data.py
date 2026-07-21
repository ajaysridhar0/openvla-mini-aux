import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.stage_release_data import (
    BODY_PART_ORDER,
    ENVIRONMENT_ALIASES,
    LAYOUT_ATTRIBUTE,
    NORMALIZED_LAYOUT,
    normalize_hdf5,
    private_attribute_hits,
    stage_file,
    stage_manifest_row,
    verify_hdf5,
)


class StageReleaseDataTest(unittest.TestCase):
    def make_file(self, path: Path, actions: np.ndarray) -> None:
        env_args = {
            "env_name": "PnPCounterToSink",
            "env_kwargs": {
                "controller_configs": {
                    "type": "HYBRID_MOBILE_BASE",
                    "composite_controller_specific_configs": {},
                }
            }
        }
        with h5py.File(path, "w") as output:
            data = output.create_group("data")
            data.attrs["env_args"] = json.dumps(env_args)
            demo = data.create_group("demo_0")
            demo.create_dataset("actions", data=actions)
            demo.attrs["ep_meta"] = json.dumps(
                {
                    "object_cfgs": [
                        {
                            "info": {
                                "mjcf_path": "/iliad2/u/private-user/project/robocasa/"
                                "models/assets/objects/mug/model.xml"
                            }
                        }
                    ]
                }
            )
            demo.attrs["model_file"] = (
                '<mujoco><asset><mesh file="/sailhome/private-user/project/robosuite/'
                'models/assets/robots/arm.stl"/></asset></mujoco>'
            )

    def test_non_panda_actions_are_losslessly_permuted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jaco.hdf5"
            original = np.arange(24, dtype=np.float32).reshape(2, 12)
            self.make_file(path, original)

            self.assertEqual(normalize_hdf5(path, "JacoOmron"), 1)

            with h5py.File(path, "r") as staged:
                data = staged["data"]
                actions = data["demo_0/actions"][...]
                expected = original[..., [0, 1, 2, 3, 4, 5, 10, 6, 7, 8, 9, 11]]
                np.testing.assert_array_equal(actions, expected)
                self.assertEqual(data.attrs[LAYOUT_ATTRIBUTE], NORMALIZED_LAYOUT)
                env_args = json.loads(data.attrs["env_args"])
                ordering = env_args["env_kwargs"]["controller_configs"][
                    "composite_controller_specific_configs"
                ]["body_part_ordering"]
                self.assertEqual(ordering, BODY_PART_ORDER)
                self.assertEqual(
                    env_args["env_name"],
                    ENVIRONMENT_ALIASES["PnPCounterToSink"],
                )
                self.assertIn("<ROBOCASA>/models/assets", data["demo_0"].attrs["ep_meta"])
                self.assertIn("<ROBOSUITE>/models/assets", data["demo_0"].attrs["model_file"])
                self.assertEqual(private_attribute_hits(staged), [])

    def test_panda_actions_stay_unchanged_and_gripper_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panda.hdf5"
            original = np.arange(12, dtype=np.float32).reshape(1, 12)
            self.make_file(path, original)

            normalize_hdf5(path, "PandaOmron")

            with h5py.File(path, "r") as staged:
                np.testing.assert_array_equal(
                    staged["data/demo_0/actions"][...], original
                )
                env_args = json.loads(staged["data"].attrs["env_args"])
                self.assertEqual(
                    env_args["env_kwargs"]["gripper_types"], "Robotiq85Gripper"
                )

    def test_staging_never_modifies_the_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.hdf5"
            destination = root / "release" / "data.hdf5"
            original = np.arange(12, dtype=np.float32).reshape(1, 12)
            self.make_file(source, original)

            stage_file(source, destination, "JacoOmron", expected_demos=1)

            with (
                h5py.File(source, "r") as archival,
                h5py.File(destination, "r") as staged,
            ):
                np.testing.assert_array_equal(
                    archival["data/demo_0/actions"][...], original
                )
                self.assertNotIn(LAYOUT_ATTRIBUTE, archival["data"].attrs)
                self.assertEqual(
                    staged["data"].attrs[LAYOUT_ATTRIBUTE], NORMALIZED_LAYOUT
                )

    def test_final_manifest_size_is_not_treated_as_source_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path("mg/JacoOmron/PnPCounterToSink/data.hdf5")
            source = root / "source" / relative
            source.parent.mkdir(parents=True)
            self.make_file(source, np.arange(12, dtype=np.float32).reshape(1, 12))
            row = {
                "relative_path": relative.as_posix(),
                "demonstrations": "1",
                "bytes": "1",
            }

            stage_manifest_row(row, root / "source", root / "release")

            self.assertTrue((root / "release" / relative).is_file())

    def test_verification_rejects_private_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.hdf5"
            self.make_file(path, np.arange(12, dtype=np.float32).reshape(1, 12))
            normalize_hdf5(path, "PandaOmron")
            with h5py.File(path, "r+") as output:
                output["data/demo_0"].attrs["scratch_path"] = "/iliad/u/user/data"

            with self.assertRaisesRegex(ValueError, "exposes private paths"):
                verify_hdf5(path, expected_demos=1)


if __name__ == "__main__":
    unittest.main()
