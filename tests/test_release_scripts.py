import argparse
import unittest
from pathlib import Path

from barx.benchmark import EMBODIMENTS, evaluation_scene_config
from scripts import build_rlds, evaluate, train


class DatasetReleaseScriptTest(unittest.TestCase):
    def test_paper_sets_resolve_to_checkpoint_compatible_dataset_names(self):
        self.assertEqual(build_rlds.stored_dataset_name("xp_900", "pnp", None), "mg_pnp_lite")
        self.assertEqual(build_rlds.stored_dataset_name("xp_3k", "flip_mug", None), "mg_flip_mug")
        self.assertEqual(build_rlds.stored_dataset_name("sp_900", "pnp", "panda_og"), "mg_panda_og_pnp")
        self.assertEqual(build_rlds.stored_dataset_name("target_50", "pnp", "jaco"), "jaco_pnp")


class TrainingLauncherTest(unittest.TestCase):
    def args(self, **updates):
        values = {
            "stage": "adapt",
            "prior": "xp_900",
            "target": "panda",
            "task": "pnp",
            "method": "ecot",
            "data_root": Path("/data/rlds"),
            "base_vlm": Path("/models/minivla"),
            "checkpoint": Path("/models/prior.pt"),
            "run_root": Path("/runs"),
            "max_steps": 3_000,
            "save_interval": 1_000,
            "gpus": 8,
            "wandb_entity": None,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def test_xp_adaptation_preserves_statistics_and_transform_aliases(self):
        command = train.build_command(self.args())
        self.assertIn("panda_xp_900_pnp", command)
        self.assertIn("xp_900_pnp_action_tokenizer", command)
        self.assertIn('{"panda_pnp":"mg_pnp_lite"}', command)
        self.assertIn("ecot", command)

    def test_target_only_and_same_embodiment_runs_start_from_base_vlm(self):
        target_only = train.build_command(self.args(prior="none", checkpoint=None))
        same_embodiment = train.build_command(self.args(prior="sp_900", checkpoint=None))
        self.assertNotIn("--pretrained_checkpoint", target_only)
        self.assertNotIn("--pretrained_checkpoint", same_embodiment)
        self.assertIn("panda_pnp", target_only)
        self.assertIn("panda_sp_900_pnp", same_embodiment)

    def test_xp_adaptation_requires_selected_prior_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires --checkpoint"):
            train.build_command(self.args(checkpoint=None))

    def test_global_batch_size_cannot_silently_change(self):
        with self.assertRaisesRegex(ValueError, "divisor"):
            train.build_command(self.args(gpus=7))


class EvaluationLauncherTest(unittest.TestCase):
    def test_paper_protocol_defaults(self):
        args = argparse.Namespace(
            checkpoint=Path("/models/model.pt"),
            embodiment="panda_og",
            task="pnp_sink_to_counter",
            inference_representation="end_effector_trace",
            unnorm_key="mg_pnp",
            episodes=100,
            start_seed=1000,
            max_steps=None,
            rollout_dir=None,
            use_wandb=False,
        )
        command = evaluate.command(args)
        self.assertIn("PandaGripper", command)
        self.assertIn("barx_panda_agentview", command)
        self.assertIn("BARXPnPSinkToCounter", command)
        self.assertIn("650", command)
        self.assertIn("8", command)
        self.assertIn("end_effector_trace", command)

    def test_every_embodiment_has_an_automatic_camera_pairing(self):
        self.assertEqual(
            set(EMBODIMENTS),
            {"iiwa", "kinova3", "ur5e", "panda", "panda_og", "jaco"},
        )
        for name, spec in EMBODIMENTS.items():
            with self.subTest(name=name):
                self.assertEqual(spec.camera, f"barx_{'panda' if name == 'panda_og' else name}_agentview")

    def test_paper_scene_filters_are_frozen(self):
        standard = evaluation_scene_config("pnp_counter_to_sink", "panda")
        panda_og_sink = evaluation_scene_config("pnp_sink_to_counter", "panda_og")
        self.assertEqual(len(standard["layout_and_style_ids"]), 32)
        self.assertEqual(len(panda_og_sink["layout_and_style_ids"]), 29)
        self.assertNotIn((8, 3), standard["layout_and_style_ids"])
        self.assertFalse(any(style == 4 for _, style in panda_og_sink["layout_and_style_ids"]))


if __name__ == "__main__":
    unittest.main()
