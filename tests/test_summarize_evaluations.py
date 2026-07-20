import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_evaluations import collect, write_csv


class EvaluationSummaryTest(unittest.TestCase):
    def test_collects_complete_and_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "task" / "panda" / "step-1000"
            failed = root / "task" / "jaco" / "step-1000"
            for run, embodiment, status in (
                (complete, "panda", "complete"),
                (failed, "jaco", "failed"),
            ):
                run.mkdir(parents=True)
                (run / "config.json").write_text(
                    json.dumps(
                        {
                            "embodiment": embodiment,
                            "pretrained_checkpoint": "/models/model.pt",
                            "task": "XPnPCounterToSink",
                        }
                    )
                )
                (run / "summary.json").write_text(
                    json.dumps(
                        {
                            "status": status,
                            "episodes": 1,
                            "successes": int(status == "complete"),
                            "success_rate": float(status == "complete"),
                        }
                    )
                )

            rows = collect(root)
            output = root / "results.csv"
            write_csv(output, rows)
            with output.open() as stream:
                saved = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["status"] for row in rows}, {"complete", "failed"})
        self.assertEqual(len(saved), 2)


if __name__ == "__main__":
    unittest.main()
