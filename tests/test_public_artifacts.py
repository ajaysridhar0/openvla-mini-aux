import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.download_public_artifacts import (
    MANIFEST_PATH,
    build_parser,
    downloads,
    main,
)


class PublicArtifactTest(unittest.TestCase):
    def test_artifact_root_keeps_runtime_layout_outside_checkout(self):
        args = build_parser().parse_args(
            ["--artifact-root", "/artifacts", "--skip-runtime-cache"]
        )
        specs, hf_home = downloads(args)
        by_repo = {spec.repo_id: spec for spec in specs}
        self.assertEqual(
            by_repo["ajaysri/robocasa-x-xp900-pnp"].local_dir,
            Path("/artifacts/data/mg_pnp_lite"),
        )
        self.assertEqual(
            by_repo["ajaysri/robocasa-x-xp900-pnp-vq-extra-action-tokenizer"].local_dir,
            Path("/artifacts/vq/mg_pnp_lite"),
        )
        self.assertEqual(hf_home, Path("/artifacts/hf"))

    def test_every_remote_snapshot_has_an_immutable_revision(self):
        args = build_parser().parse_args(["--artifact-root", "/artifacts"])
        specs, _ = downloads(args)
        self.assertEqual(len(specs), 6)
        for spec in specs:
            self.assertEqual(len(spec.revision), 40)
            int(spec.revision, 16)

    def test_base_download_excludes_historical_checkpoints_and_wandb(self):
        args = build_parser().parse_args(
            ["--artifact-root", "/artifacts", "--skip-runtime-cache"]
        )
        specs, _ = downloads(args)
        base = next(spec for spec in specs if spec.name == "BARX base VLM")
        self.assertEqual(
            base.allow_patterns, ("config.json", "checkpoints/latest-checkpoint.pt")
        )
        self.assertIn("checkpoints/latest-checkpoint.pt", base.expected_files)

    def test_manifest_is_machine_readable_and_versioned(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(len(manifest["runtime_dependencies"]), 3)

    def test_runtime_code_uses_manifest_revisions(self):
        root = MANIFEST_PATH.parents[1]
        runtime_source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                root / "policy/prismatic/models/backbones/llm/qwen25.py",
                root / "policy/prismatic/models/backbones/vision/dinosiglip_vit.py",
            )
        )
        for dependency in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))[
            "runtime_dependencies"
        ]:
            self.assertIn(dependency["revision"], runtime_source)

    def test_dry_run_never_imports_hugging_face(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "sys.argv",
                [
                    "download_public_artifacts.py",
                    "--artifact-root",
                    directory,
                    "--dry-run",
                ],
            ),
            mock.patch.dict("sys.modules", {"huggingface_hub": None}),
        ):
            main()


if __name__ == "__main__":
    unittest.main()
