import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from barx.auth import resolve_hf_token


class AuthenticationContractTest(unittest.TestCase):
    def test_public_artifacts_need_no_token(self):
        self.assertIsNone(resolve_hf_token(None))

    def test_token_can_come_from_file_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / "token"
            token_file.write_text("file-token\n")
            self.assertEqual(resolve_hf_token(token_file), "file-token")
        with mock.patch.dict(os.environ, {"BARX_TEST_TOKEN": "env-token"}):
            self.assertEqual(resolve_hf_token("BARX_TEST_TOKEN"), "env-token")

    def test_missing_token_environment_variable_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "is not set"):
            resolve_hf_token("BARX_INTENTIONALLY_MISSING_TOKEN")


class SimulatorRegistrationTest(unittest.TestCase):
    def test_registration_import_is_part_of_package_initialization(self):
        package_init = Path("robocasa_x/robocasa/models/__init__.py").read_text()
        self.assertIn("robocasa.models.compositional_robots import", package_init)

    def test_custom_mobile_robots_register_on_normal_import(self):
        if importlib.util.find_spec("robosuite") is None:
            self.skipTest("full simulator dependencies are not installed")
        import robocasa.models  # noqa: F401
        from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

        self.assertIn("IIWAOmron", REGISTERED_ROBOTS)
        self.assertIn("JacoOmron", REGISTERED_ROBOTS)


class DependencyContractTest(unittest.TestCase):
    def test_tensorflow_metadata_matches_locked_protobuf_runtime(self):
        from tensorflow_metadata.proto.v0 import schema_pb2

        self.assertEqual(schema_pb2.Schema.DESCRIPTOR.file.name, "tensorflow_metadata/proto/v0/schema.proto")


if __name__ == "__main__":
    unittest.main()
