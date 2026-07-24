import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from barx.raw_data import RawSubset, all_subsets, selected_rows
from scripts import build_raw_subsets, download_raw_data
from scripts.stage_release_data import LAYOUT_ATTRIBUTE, NORMALIZED_LAYOUT
from scripts.verify_raw_data import verify_dataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset" / "manifest.csv"
PUBLIC_DATASET = ROOT / "configs" / "raw_dataset.json"


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

    def test_public_dataset_is_immutable_and_versioned(self):
        config = json.loads(PUBLIC_DATASET.read_text())
        self.assertEqual(config["repo_id"], "ajaysri/barx-raw-hdf5")
        self.assertEqual(config["repo_type"], "dataset")
        self.assertEqual(config["tag"], "v1.0.0")
        self.assertEqual(
            config["collection"],
            "ajaysri/barx-raw-hdf5-data-6a61b2d60e2a7ca90b75fb68",
        )
        self.assertEqual(len(config["revision"]), 40)
        int(config["revision"], 16)

    def test_raw_subsets_match_the_24_public_rlds_datasets(self):
        with MANIFEST.open(newline="") as stream:
            master_rows = list(csv.DictReader(stream))
        subsets = all_subsets()
        self.assertEqual(len(subsets), 24)
        self.assertEqual(len({subset.slug for subset in subsets}), 24)
        self.assertEqual(
            sum(len(selected_rows(master_rows, subset)) for subset in subsets), 276
        )
        self.assertEqual(
            {
                row["relative_path"]
                for subset in subsets
                for row in selected_rows(master_rows, subset)
            },
            {row["relative_path"] for row in master_rows},
        )

    def test_checked_in_subset_manifests_are_current(self):
        build_raw_subsets.build(check=True)

    def test_selective_downloader_uses_pinned_anonymous_paths(self):
        args = download_raw_data.build_parser().parse_args(
            [
                "--dataset",
                "target_50",
                "--target",
                "panda",
                "--task",
                "pnp",
                "--output-dir",
                "/tmp/barx-target-panda-pnp",
            ]
        )
        subset = download_raw_data.selection(args)
        spec, manifest, rows = download_raw_data.download_spec(args, subset)
        config = json.loads(PUBLIC_DATASET.read_text())
        self.assertEqual(subset, RawSubset("target_50", "pnp", "panda"))
        self.assertEqual(spec.repo_id, config["repo_id"])
        self.assertEqual(spec.revision, config["revision"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(int(row["demonstrations"]) for row in rows), 100)
        self.assertEqual(manifest.name, "robocasa-x-target-panda-pnp.csv")
        self.assertIn("manifest.csv", spec.allow_patterns)
        self.assertTrue(
            all(
                row["relative_path"] in spec.allow_patterns
                for row in rows
            )
        )

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
