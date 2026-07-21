import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock
from zipfile import ZipFile


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "robocasa_x"
    / "robocasa"
    / "scripts"
    / "download_kitchen_assets.py"
)
SPEC = importlib.util.spec_from_file_location("barx_download_kitchen_assets", SCRIPT)
ASSETS = importlib.util.module_from_spec(SPEC)


class _FakeTqdm:
    def __init__(self, *args, **kwargs):
        self.n = 0
        self.total = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, amount):
        self.n += amount


termcolor = ModuleType("termcolor")
termcolor.colored = lambda text, *_args, **_kwargs: text
tqdm = ModuleType("tqdm")
tqdm.tqdm = _FakeTqdm
robocasa = ModuleType("robocasa")
robocasa.__path__ = ["/tmp/robocasa"]
with mock.patch.dict(
    "sys.modules", {"termcolor": termcolor, "tqdm": tqdm, "robocasa": robocasa}
):
    SPEC.loader.exec_module(ASSETS)


class KitchenAssetDownloaderTest(unittest.TestCase):
    def test_all_release_archives_have_sizes_and_sha256(self):
        self.assertEqual(len(ASSETS.DOWNLOAD_ASSET_REGISTRY), 4)
        for config in ASSETS.DOWNLOAD_ASSET_REGISTRY.values():
            self.assertGreater(config["size_bytes"], 0)
            self.assertEqual(len(config["sha256"]), 64)
            int(config["sha256"], 16)

    def test_archive_verification_accepts_exact_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset.zip"
            path.write_bytes(b"BARX")
            ASSETS._verify_archive(path, 4, hashlib.sha256(b"BARX").hexdigest())

    def test_safe_extraction_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.zip"
            with ZipFile(archive, "w") as stream:
                stream.writestr("../escaped.txt", "unsafe")
            with self.assertRaisesRegex(RuntimeError, "unsafe archive path"):
                ASSETS._safe_extract(archive, root / "extract")

    def test_three_failed_downloads_raise(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                ASSETS.urllib.request, "urlretrieve", side_effect=OSError("offline")
            ) as download,
        ):
            with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                ASSETS.download_and_extract_zip(
                    "https://example.com/assets.zip",
                    Path(directory) / "assets",
                    1,
                    "0" * 64,
                )
            self.assertEqual(download.call_count, 3)


if __name__ == "__main__":
    unittest.main()
