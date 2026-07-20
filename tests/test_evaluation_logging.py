import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from barx.evaluation_logging import (
    EvaluationRun,
    checkpoint_label,
    create_run_directory,
)


@dataclass
class ExampleConfig:
    checkpoint: Path


class EvaluationLoggingTest(unittest.TestCase):
    def test_checkpoint_labels_are_readable_and_safe(self):
        self.assertEqual(checkpoint_label("run-step-050000-epoch-2.pt"), "step-50000")
        self.assertEqual(checkpoint_label("My checkpoint!.pt"), "My-checkpoint")

    def test_run_directories_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            template = Path(directory) / "STEP"
            first = create_run_directory(template, "model-step-001000-epoch-1.pt")
            second = create_run_directory(template, "model-step-001000-epoch-1.pt")
        self.assertEqual(first.name, "step-1000")
        self.assertEqual(second.name, "step-1000-run-002")

    def test_run_writes_machine_readable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            with EvaluationRun(run_dir, ExampleConfig(Path("model.pt"))) as run:
                run.write("hello\n")
                run.record_episode({"episode": 0, "success": True})
                run.finalize({"success_rate": 1.0}, status="complete")
            config = json.loads((run_dir / "config.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            episodes = (run_dir / "episodes.jsonl").read_text().splitlines()

        self.assertEqual(config["checkpoint"], "model.pt")
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(json.loads(episodes[0])["episode"], 0)


if __name__ == "__main__":
    unittest.main()
