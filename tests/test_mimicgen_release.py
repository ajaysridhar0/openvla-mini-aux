import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from barx.mimicgen_release import (
    embodiment_names,
    generated_inventory,
    registry,
    resolved_config,
    source_inventory,
    task_names,
)


ROOT = Path(__file__).resolve().parents[1]
MIMICGEN_ROOT = ROOT / "third_party" / "mimicgen"


def write_fixture(path: Path, *, prepared: int = 0, normalized: bool = False) -> None:
    with h5py.File(path, "w") as output:
        data = output.create_group("data")
        data.attrs["env_args"] = json.dumps(
            {"env_name": "XFlipMugUpright", "env_kwargs": {}}
        )
        if normalized:
            data.attrs["barx_action_layout"] = (
                "arm_gripper_base_torso_mode_v1"
            )
        for index in range(2):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=np.zeros((3, 12)))
            demo.create_dataset("states", data=np.zeros((3, 8 + index)))
            if index < prepared:
                demo.create_group("datagen_info")


class MimicGenReleaseTest(unittest.TestCase):
    def test_registry_resolves_all_paper_combinations(self):
        self.assertEqual(len(task_names()), 4)
        self.assertEqual(len(embodiment_names()), 6)
        for task in task_names():
            for embodiment in embodiment_names():
                with self.subTest(task=task, embodiment=embodiment):
                    config = resolved_config(
                        source=Path("/tmp/source.hdf5"),
                        task=task,
                        embodiment=embodiment,
                        output_dir=Path("/tmp/generated"),
                        seed=7,
                        successes=1,
                        max_attempts=10,
                        source_demos=5,
                    )
                    self.assertEqual(config["experiment"]["seed"], 7)
                    self.assertEqual(
                        config["experiment"]["generation"]["max_attempts"], 10
                    )
                    self.assertEqual(
                        config["experiment"]["task"]["interface"],
                        registry()["tasks"][task]["interface"],
                    )

    def test_invalid_generation_bounds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_attempts"):
            resolved_config(
                source=Path("/tmp/source.hdf5"),
                task="flip_mug_upright",
                embodiment="panda",
                output_dir=Path("/tmp/generated"),
                seed=0,
                successes=2,
                max_attempts=1,
                source_demos=1,
            )

    def test_source_and_generated_inventories_are_machine_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.hdf5"
            generated = Path(directory) / "generated.hdf5"
            write_fixture(source, prepared=1)
            write_fixture(generated, normalized=True)

            self.assertEqual(
                source_inventory(source),
                {
                    "demonstrations": 2,
                    "prepared_demonstrations": 1,
                    "action_widths": [12],
                    "environment": "XFlipMugUpright",
                },
            )
            inventory = generated_inventory(generated)
            self.assertEqual(inventory["demonstrations"], 2)
            self.assertEqual(inventory["action_widths"], [12])
            self.assertEqual(
                inventory["action_layout"],
                "arm_gripper_base_torso_mode_v1",
            )
            self.assertEqual(inventory["state_shapes"], [(3, 8), (3, 9)])
            self.assertEqual(len(inventory["sha256"]), 64)

    def test_vendored_snapshot_is_licensed_and_has_no_private_paths(self):
        self.assertTrue((MIMICGEN_ROOT / "LICENSE").is_file())
        self.assertIn(
            "NVIDIA License",
            (MIMICGEN_ROOT / "LICENSE").read_text(),
        )
        markers = ("/iliad", "/sailhome", "/home/", "/users/")
        hits = []
        for path in MIMICGEN_ROOT.rglob("*"):
            if path.suffix not in {".py", ".json", ".toml", ".md"}:
                continue
            text = path.read_text(errors="replace").lower()
            if any(marker in text for marker in markers):
                hits.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(hits, [])

    def test_vendored_generator_is_bounded_and_uses_canonical_layout(self):
        generator = (
            MIMICGEN_ROOT / "mimicgen" / "scripts" / "generate_dataset.py"
        ).read_text()
        interface = (
            MIMICGEN_ROOT
            / "mimicgen"
            / "env_interfaces"
            / "robosuite.py"
        ).read_text()
        waypoint = (
            MIMICGEN_ROOT / "mimicgen" / "datagen" / "waypoint.py"
        ).read_text()
        self.assertIn("max_attempts", generator)
        self.assertIn("num_attempts >= max_attempts", generator)
        self.assertIn("return action[6:7]", interface)
        self.assertNotIn('if "panda" in robot_type.lower()', waypoint)


if __name__ == "__main__":
    unittest.main()
