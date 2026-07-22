import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.stage_release_data import LAYOUT_ATTRIBUTE, NORMALIZED_LAYOUT
from scripts.verify_raw_data import verify_dataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset" / "manifest.csv"


class RawDataTest(unittest.TestCase):
    def make_normalized_hdf5(self, path: Path) -> None:
        env_args = {
            "env_name": "XPnPCounterToSink",
            "env_kwargs": {
                "controller_configs": {"composite_controller_specific_configs": {}}
            },
        }
        with h5py.File(path, "w") as output:
            data = output.create_group("data")
            data.attrs[LAYOUT_ATTRIBUTE] = NORMALIZED_LAYOUT
            data.attrs["env_args"] = json.dumps(env_args)
            demo = data.create_group("demo_0")
            demo.create_dataset("actions", data=np.zeros((2, 12)))
            demo.attrs["ep_meta"] = json.dumps({"lang": "move the object"})

    def test_public_manifest_is_complete_and_checksummed(self):
        with MANIFEST.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 240)
        self.assertEqual(sum(int(row["demonstrations"]) for row in rows), 23_400)
        for row in rows:
            self.assertEqual(len(row["sha256"]), 64)
            int(row["sha256"], 16)

    def test_verifier_accepts_a_matching_normalized_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path("human/PandaOmron/PnPCounterToSink/demo.hdf5")
            path = root / relative
            path.parent.mkdir(parents=True)
            self.make_normalized_hdf5(path)

            manifest = root / "manifest.csv"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["relative_path", "bytes", "sha256", "demonstrations"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": relative.as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": digest,
                        "demonstrations": 1,
                    }
                )

            self.assertEqual(
                verify_dataset(root, manifest, workers=1), (1, 1, path.stat().st_size)
            )

    def test_verifier_rejects_checksum_mismatch_without_rewriting_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "demo.hdf5"
            self.make_normalized_hdf5(path)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["relative_path", "bytes", "sha256", "demonstrations"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "relative_path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": "0" * 64,
                        "demonstrations": 1,
                    }
                )
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_dataset(root, manifest, workers=1)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
