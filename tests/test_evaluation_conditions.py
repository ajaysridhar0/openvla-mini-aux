import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from barx.evaluation_conditions import (
    load_bundle,
    make_entry,
    materialize_ep_meta,
    materialize_model_xml,
    model_sha256,
    portable_ep_meta,
    portable_model_xml,
    stable_replay_metadata,
    write_bundle,
)


class EvaluationConditionTest(unittest.TestCase):
    def test_episode_metadata_paths_are_portable(self):
        metadata = {
            "object_cfgs": [
                {"mjcf_path": "/tmp/install/robocasa/models/assets/object.xml"}
            ],
            "texture": "/other/install/robosuite/models/texture.png",
        }
        portable = portable_ep_meta(metadata)
        self.assertEqual(
            portable["object_cfgs"][0]["mjcf_path"],
            "<ROBOCASA>/models/assets/object.xml",
        )
        restored = materialize_ep_meta(
            portable,
            {
                "robocasa": Path("/new/robocasa"),
                "robosuite": Path("/new/robosuite"),
            },
        )
        self.assertEqual(
            restored["texture"], "/new/robosuite/models/texture.png"
        )

    def entry(self, episode, seed, state):
        return make_entry(
            episode=episode,
            seed=seed,
            ep_meta={
                "layout_id": 4,
                "style_id": episode,
                "lang": "pick the carrot",
                "object_cfgs": [],
            },
            model_xml='<mujoco><mesh file="/machine/robocasa/assets/a.stl"/></mujoco>',
            policy_start_state=state,
        )

    def test_round_trip_validates_metadata_and_states(self):
        states = [np.arange(5, dtype=np.float64), np.arange(5, dtype=np.float64) + 1]
        entries = [self.entry(index, 1000 + index, state) for index, state in enumerate(states)]
        models = [
            '<mujoco><mesh file="/machine/robocasa/assets/a.stl"/></mujoco>'
        ] * 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(
                root,
                task="pnp_counter_to_sink",
                embodiment="panda",
                entries=entries,
                states=states,
                model_xmls=models,
                provenance={"source": "test"},
            )
            payload, loaded_states, loaded_models = load_bundle(
                root,
                task="pnp_counter_to_sink",
                embodiment="panda",
            )
        self.assertEqual(payload["episode_count"], 2)
        np.testing.assert_array_equal(loaded_states, np.stack(states))
        self.assertEqual(loaded_models, [portable_model_xml(model) for model in models])

    def test_corrupt_metadata_is_rejected(self):
        state = np.arange(3, dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path, _ = write_bundle(
                root,
                task="flip_mug_upright",
                embodiment="jaco",
                entries=[self.entry(0, 1000, state)],
                states=[state],
                model_xmls=[
                    '<mujoco><mesh file="/machine/robocasa/assets/a.stl"/></mujoco>'
                ],
                provenance={},
            )
            payload = json.loads(json_path.read_text())
            payload["episodes"][0]["instruction"] = "changed"
            payload["episodes"][0]["ep_meta"]["lang"] = "changed"
            json_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "Corrupt metadata"):
                load_bundle(root, task="flip_mug_upright", embodiment="jaco")

    def test_model_hash_ignores_install_prefix(self):
        first = '<mesh file="/a/robocasa/assets/object.stl"/>'
        second = '<mesh file="/b/robocasa/assets/object.stl"/>'
        self.assertEqual(model_sha256(first), model_sha256(second))

    def test_model_xml_paths_are_portable(self):
        xml = '<mesh file="/old/robocasa/models/a.stl"/>'
        portable = portable_model_xml(xml)
        self.assertEqual(portable, '<mesh file="<ROBOCASA>/models/a.stl"/>')
        self.assertEqual(
            materialize_model_xml(
                portable,
                {
                    "robocasa": Path("/new/robocasa"),
                    "robosuite": Path("/new/robosuite"),
                },
            ),
            '<mesh file="/new/robocasa/models/a.stl"/>',
        )

    def test_replay_metadata_ignores_mutated_object_placements(self):
        first = {"layout_id": 7, "object_cfgs": [{"placement": "sampled"}]}
        second = {"layout_id": 7, "object_cfgs": [{"placement": "resolved"}]}
        self.assertEqual(stable_replay_metadata(first), stable_replay_metadata(second))


if __name__ == "__main__":
    unittest.main()
