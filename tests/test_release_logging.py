import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseCommandLoggerTest(unittest.TestCase):
    def test_command_output_and_metadata_are_append_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "command.sh"
            script.write_text(
                "printf 'standard output\\n'\n"
                "printf 'standard error\\n' >&2\n"
            )
            command = [
                sys.executable,
                str(ROOT / "scripts" / "log_release_command.py"),
                "--run-dir",
                str(root / "run"),
                "--command-id",
                "S01-C01",
                "--section",
                "install",
                "--cwd",
                str(ROOT),
                "--script",
                str(script),
            ]
            completed = subprocess.run(command, check=True, capture_output=True)
            self.assertIn(b"standard output", completed.stdout)
            self.assertIn(b"standard error", completed.stderr)
            record = json.loads(
                (root / "run" / "commands.jsonl").read_text().strip()
            )
            self.assertEqual(record["exit_code"], 0)
            self.assertEqual(record["status"], "PASS")
            self.assertEqual(len(record["stdout_sha256"]), 64)

            duplicate = subprocess.run(command, capture_output=True)
            self.assertNotEqual(duplicate.returncode, 0)
            self.assertIn(b"evidence already exists", duplicate.stderr)


if __name__ == "__main__":
    unittest.main()
