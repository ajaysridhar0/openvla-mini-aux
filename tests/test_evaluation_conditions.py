import copy
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
    project_target_center,
    stable_replay_metadata,
    validate_condition_bundle_for_evaluation,
    validate_condition_semantics,
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
        self.assertEqual(restored["texture"], "/new/robosuite/models/texture.png")

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
        entries = [
            self.entry(index, 1000 + index, state) for index, state in enumerate(states)
        ]
        models = ['<mujoco><mesh file="/machine/robocasa/assets/a.stl"/></mujoco>'] * 2
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
        self.assertEqual(len(loaded_states), len(states))
        for loaded, expected in zip(loaded_states, states):
            np.testing.assert_array_equal(loaded, expected)
        self.assertEqual(loaded_models, [portable_model_xml(model) for model in models])

    def test_round_trip_supports_variable_length_simulator_states(self):
        states = [
            np.arange(3, dtype=np.float64),
            np.arange(7, dtype=np.float64),
        ]
        entries = [
            self.entry(index, 1000 + index, state) for index, state in enumerate(states)
        ]
        models = ['<mujoco><mesh file="/machine/robocasa/assets/a.stl"/></mujoco>'] * 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(
                root,
                task="pnp_sink_to_counter",
                embodiment="panda_og",
                entries=entries,
                states=states,
                model_xmls=models,
                provenance={"source": "test"},
            )
            _, loaded_states, _ = load_bundle(
                root,
                task="pnp_sink_to_counter",
                embodiment="panda_og",
            )
        self.assertEqual([len(state) for state in loaded_states], [3, 7])
        for loaded, expected in zip(loaded_states, states):
            np.testing.assert_array_equal(loaded, expected)

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

    def test_target_projection_detects_out_of_frame_object(self):
        model_xml = """
        <mujoco>
          <worldbody>
            <body name="camera_base">
              <camera name="policy_camera" fovy="90"/>
            </body>
            <body name="obj_main">
              <freejoint name="obj_joint0"/>
            </body>
          </worldbody>
        </mujoco>
        """
        state = np.concatenate(
            [
                [0.0],
                [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                np.zeros(6),
            ]
        )
        projection = project_target_center(
            {"condition_id": "synthetic"},
            state,
            model_xml,
            camera_name="policy_camera",
            image_width=320,
            image_height=180,
        )
        self.assertTrue(projection["center_in_frame"])
        self.assertAlmostEqual(projection["pixel_x"], 160.0)
        self.assertAlmostEqual(projection["pixel_y"], 90.0)

        state[1] = 3.0
        projection = project_target_center(
            {"condition_id": "synthetic"},
            state,
            model_xml,
            camera_name="policy_camera",
            image_width=320,
            image_height=180,
        )
        self.assertFalse(projection["center_in_frame"])

    def test_checked_in_pnp_targets_use_the_paper_object_set(self):
        conditions = Path(__file__).resolve().parents[1] / "evaluation" / "conditions"
        for task in ("pnp_counter_to_sink", "pnp_sink_to_counter"):
            for metadata_path in (conditions / task).glob("*.json"):
                payload = json.loads(metadata_path.read_text())
                for entry in payload["episodes"]:
                    with self.subTest(
                        task=task,
                        embodiment=payload["embodiment"],
                        seed=entry["seed"],
                    ):
                        validate_condition_semantics(entry, task=task)

    def test_embodiments_share_the_same_condition_seeds_per_task(self):
        conditions = Path(__file__).resolve().parents[1] / "evaluation" / "conditions"
        for task in (
            "pnp_counter_to_sink",
            "pnp_sink_to_counter",
            "turn_on_sink_faucet",
            "flip_mug_upright",
        ):
            seed_lists = {}
            for metadata_path in sorted((conditions / task).glob("*.json")):
                payload = json.loads(metadata_path.read_text())
                seed_lists[payload["embodiment"]] = [
                    entry["seed"] for entry in payload["episodes"]
                ]
            reference_embodiment, reference_seeds = next(iter(seed_lists.items()))
            for embodiment, seeds in seed_lists.items():
                with self.subTest(task=task, embodiment=embodiment):
                    self.assertEqual(
                        seeds,
                        reference_seeds,
                        f"{embodiment} differs from {reference_embodiment}",
                    )

    def test_semantic_validation_rejects_wrong_object_category_and_split(self):
        conditions = Path(__file__).resolve().parents[1] / "evaluation" / "conditions"
        payload, _, _ = load_bundle(
            conditions,
            task="pnp_counter_to_sink",
            embodiment="panda",
        )
        entry = copy.deepcopy(payload["episodes"][0])
        target = next(
            cfg for cfg in entry["ep_meta"]["object_cfgs"] if cfg["name"] == "obj"
        )

        target["info"]["cat"] = "potato"
        entry["instruction"] = (
            "pick the potato from the counter and place it in the sink"
        )
        with self.assertRaisesRegex(ValueError, "not in obj_set1"):
            validate_condition_semantics(entry, task="pnp_counter_to_sink")

        entry = copy.deepcopy(payload["episodes"][0])
        target = next(
            cfg for cfg in entry["ep_meta"]["object_cfgs"] if cfg["name"] == "obj"
        )
        target["info"]["split"] = "B"
        with self.assertRaisesRegex(ValueError, "object split A"):
            validate_condition_semantics(entry, task="pnp_counter_to_sink")

    def test_release_excludes_showcased_out_of_frame_seed(self):
        evaluation = Path(__file__).resolve().parents[1] / "evaluation"
        historical = json.loads(
            (evaluation / "HISTORICAL_VISIBILITY_AUDIT.json").read_text()
        )
        bad_seed = next(
            condition
            for condition in historical["invalid_conditions"]
            if condition["task"] == "pnp_counter_to_sink"
            and condition["embodiment"] == "panda"
            and condition["seed"] == 1001
        )
        self.assertEqual(
            bad_seed["reason"], "target center is outside the policy camera"
        )
        self.assertAlmostEqual(bad_seed["projection"]["pixel_x"], -91.0, delta=0.1)
        self.assertAlmostEqual(bad_seed["projection"]["pixel_y"], 212.9, delta=0.1)

        conditions = evaluation / "conditions"
        payload, states, model_xmls = load_bundle(
            conditions,
            task="pnp_counter_to_sink",
            embodiment="panda",
        )
        self.assertEqual(
            payload["provenance"]["condition_protocol"], "visible-target-v1"
        )
        self.assertNotIn(1001, [entry["seed"] for entry in payload["episodes"]])
        validate_condition_bundle_for_evaluation(
            payload,
            states,
            model_xmls,
            task="pnp_counter_to_sink",
            camera_name="barx_panda_agentview",
            image_width=320,
            image_height=180,
            count=100,
        )


if __name__ == "__main__":
    unittest.main()
