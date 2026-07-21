import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.download_public_artifacts import downloads, main


class PublicArtifactTest(unittest.TestCase):
    def test_xp900_pnp_downloads_use_historical_runtime_layout(self):
        args = argparse.Namespace(
            data_root=Path("/data"),
            base_vlm_dir=Path("/models/base"),
            vq_root=Path("/vq"),
            run_root=Path("/runs"),
            include_pretrain_checkpoint=False,
        )
        items = downloads(args)
        self.assertEqual(items[1][2], Path("/data/mg_pnp_lite"))
        self.assertEqual(items[2][2], Path("/vq/mg_pnp_lite"))
        self.assertEqual({item[1] for item in items}, {"model", "dataset"})

    def test_dry_run_never_calls_hugging_face(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "sys.argv",
            [
                "download_public_artifacts.py",
                "--data-root",
                f"{directory}/data",
                "--base-vlm-dir",
                f"{directory}/base",
                "--dry-run",
            ],
        ):
            main()


if __name__ == "__main__":
    unittest.main()
