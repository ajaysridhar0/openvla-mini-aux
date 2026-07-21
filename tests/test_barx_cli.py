"""Fast tests for BARX preset expansion; no simulator or model is required."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from barx.commands import (
    build_eval_command,
    build_generation_config,
    build_train_command,
    dataset_name,
    downloadable_datasets,
    infer_unnorm_key,
)
from barx.config import ConfigError, load_presets


class BarxPresetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presets = load_presets()

    def train_args(self, **overrides):
        values = {
            "stage": "pretrain",
            "split": "xp900",
            "task": "pnp",
            "representation": "joint-reps",
            "robot": None,
            "checkpoint": None,
            "base_vlm": Path("/models/base"),
            "data_root": Path("/data"),
            "run_root": Path("/runs"),
            "gpus": 8,
            "max_steps": None,
            "save_interval": None,
            "run_name": None,
            "wandb": False,
            "wandb_entity": None,
            "wandb_project": "barx",
            "hf_token_env": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_dataset_names_cover_paper_stages(self):
        self.assertEqual(dataset_name("pretrain", "xp900", "pnp", None), "robocasa-x-xp900-pnp")
        self.assertEqual(dataset_name("finetune", "xp3k", "pnp", "jaco"), "robocasa-x-xp3k-jaco-pnp-mix")
        self.assertEqual(
            dataset_name("pretrain", "sp900", "flip-mug-upright", "panda-og"),
            "robocasa-x-sp900-panda-og-flip-mug-upright",
        )
        self.assertEqual(
            dataset_name("target-only", "xp900", "turn-on-sink-faucet", "panda"),
            "robocasa-x-target-panda-turn-on-sink-faucet",
        )
        self.assertEqual(
            downloadable_datasets("finetune", "xp3k", "pnp", "jaco"),
            ["robocasa-x-xp3k-pnp", "robocasa-x-target-jaco-pnp"],
        )

    def test_joint_rep_pretrain_expands_to_released_settings(self):
        command, name = build_train_command(self.train_args(), self.presets)
        rendered = " ".join(command)
        self.assertEqual(name, "pretrain-xp900-pnp-joint-reps")
        self.assertIn("robocasa-x-xp900-pnp", command)
        self.assertIn("bbox->,low_level_motion->,ee_pose_2D->,action", command)
        self.assertIn("100000", command)
        self.assertIn("[jsonl]", command)
        self.assertNotIn("--pretrained_checkpoint", rendered)

    def test_finetune_reuses_prior_tokenizer_and_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "prior"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "model.pt"
            checkpoint.touch()
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "dataset_statistics.json").write_text("{}", encoding="utf-8")
            args = self.train_args(stage="finetune", split="xp3k", robot="jaco", checkpoint=checkpoint)
            command, _ = build_train_command(args, self.presets)
        rendered = " ".join(command)
        self.assertIn("robocasa-x-xp3k-jaco-pnp-mix", command)
        self.assertIn("robocasa-x-xp3k-pnp-vq-extra-action-tokenizer", command)
        self.assertIn('"robocasa-x-target-jaco-pnp": "robocasa-x-xp3k-pnp"', rendered)
        self.assertIn("--is_resume False", rendered)

    def test_source_robot_is_rejected_for_target_training(self):
        with self.assertRaises(ConfigError):
            build_train_command(self.train_args(stage="target-only", robot="iiwa"), self.presets)

    def test_generation_config_encodes_robot_and_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.hdf5"
            source.touch()
            args = SimpleNamespace(
                source=source,
                output=Path(temp_dir) / "out",
                environment="pnp-counter-to-sink",
                robot="panda-og",
                num_demos=7,
                seed=3,
            )
            config = build_generation_config(args, self.presets)
        self.assertEqual(config["name"], "PnPCounterToSink")
        self.assertEqual(config["experiment"]["task"]["robot"], "PandaOmron")
        self.assertEqual(config["experiment"]["task"]["gripper"], "PandaGripper")
        self.assertEqual(config["task"]["task_spec"]["stage_2"]["object_ref"], "sink")

    def test_sink_to_counter_keeps_historical_second_stage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = SimpleNamespace(
                source=Path(temp_dir) / "source.hdf5",
                output=Path(temp_dir) / "out",
                environment="pnp-sink-to-counter",
                robot="jaco",
                num_demos=1,
                seed=0,
            )
            config = build_generation_config(args, self.presets)
        stage_2 = config["task"]["task_spec"]["stage_2"]
        self.assertNotIn("num_interpolation_steps", stage_2)
        self.assertNotIn("gripper", config["experiment"]["task"])

    def test_eval_infers_statistics_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "step-001000-epoch-01-loss=0.1.pt"
            checkpoint.touch()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "vla": {"data_mix": "robocasa-x-xp900-jaco-pnp-mix"},
                        "dataset_statistics_map": {"robocasa_x_target_jaco_pnp": "robocasa_x_xp900_pnp"},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "dataset_statistics.json").write_text(
                json.dumps({"robocasa_x_target_jaco_pnp": {}, "robocasa_x_xp900_pnp": {}}), encoding="utf-8"
            )
            self.assertEqual(infer_unnorm_key(checkpoint), "robocasa_x_xp900_pnp")

            args = SimpleNamespace(
                checkpoint=checkpoint,
                environment="pnp-counter-to-sink",
                robot="jaco",
                unnorm_key=None,
                model_tag=None,
                output=None,
                run_root=Path(temp_dir) / "rollouts",
                trials=2,
                start_seed=1000,
                no_videos=True,
                max_videos=0,
                wandb=False,
                hf_token_env=None,
            )
            command, _ = build_eval_command(args, self.presets)
        self.assertIn("JacoOmron", command)
        self.assertIn("PnPCounterToSink", command)
        self.assertIn("robocasa_x_xp900_pnp", command)


if __name__ == "__main__":
    unittest.main()
