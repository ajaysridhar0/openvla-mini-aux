import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.download_public_artifacts import (
    DownloadSpec,
    MANIFEST_PATH,
    build_parser,
    download_and_verify,
    download_snapshot,
    downloads,
    main,
)


class PublicArtifactTest(unittest.TestCase):
    def setUp(self):
        self.args = build_parser().parse_args(["--artifact-root", "/artifacts"])
        self.spec = DownloadSpec(
            name="test artifact",
            repo_id="organization/repository",
            repo_type="model",
            revision="a" * 40,
            local_dir=Path("/artifacts/model"),
            allow_patterns=("config.json",),
            expected_files={},
        )

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

    def test_download_controls_have_conservative_defaults(self):
        self.assertEqual(self.args.max_workers, 1)
        self.assertEqual(self.args.etag_timeout, 60)
        self.assertEqual(self.args.retries, 6)

    def test_snapshot_receives_worker_and_timeout_arguments(self):
        self.args.max_workers = 3
        self.args.etag_timeout = 90
        snapshot_download = mock.Mock()
        download_snapshot(
            self.spec, Path("/artifacts/hf"), self.args, snapshot_download
        )
        snapshot_download.assert_called_once_with(
            repo_id=self.spec.repo_id,
            repo_type=self.spec.repo_type,
            revision=self.spec.revision,
            allow_patterns=["config.json"],
            token=False,
            max_workers=3,
            etag_timeout=90,
            local_dir=self.spec.local_dir,
        )

    def test_retry_after_header_controls_backoff(self):
        error = RuntimeError("rate limited")
        error.response = SimpleNamespace(headers={"Retry-After": "17"})
        snapshot_download = mock.Mock(side_effect=[error, None])
        with mock.patch("scripts.download_public_artifacts.time.sleep") as sleep:
            download_snapshot(
                self.spec, Path("/artifacts/hf"), self.args, snapshot_download
            )
        self.assertEqual(snapshot_download.call_count, 2)
        sleep.assert_called_once_with(17.0)

    def test_final_download_error_is_propagated(self):
        self.args.retries = 2
        error = OSError("offline")
        snapshot_download = mock.Mock(side_effect=error)
        with mock.patch("scripts.download_public_artifacts.time.sleep") as sleep:
            with self.assertRaises(OSError) as raised:
                download_snapshot(
                    self.spec, Path("/artifacts/hf"), self.args, snapshot_download
                )
        self.assertIs(raised.exception, error)
        self.assertEqual(snapshot_download.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])

    def test_checksum_failure_is_not_retried(self):
        checksum_error = RuntimeError("SHA-256 mismatch")
        snapshot_download = mock.Mock()
        with (
            mock.patch(
                "scripts.download_public_artifacts.verify_files",
                side_effect=checksum_error,
            ) as verify,
            mock.patch("scripts.download_public_artifacts.time.sleep") as sleep,
        ):
            with self.assertRaises(RuntimeError) as raised:
                download_and_verify(
                    self.spec,
                    Path("/artifacts/hf"),
                    self.args,
                    snapshot_download,
                )
        self.assertIs(raised.exception, checksum_error)
        snapshot_download.assert_called_once()
        verify.assert_called_once_with(self.spec)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
