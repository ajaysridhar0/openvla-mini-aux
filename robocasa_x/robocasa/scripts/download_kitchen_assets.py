"""Download and verify the RoboCasa kitchen assets used by BARX."""

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from termcolor import colored
from tqdm import tqdm

import robocasa


DOWNLOAD_ASSET_REGISTRY = {
    "textures": {
        "message": "Downloading environment textures",
        "url": "https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip",
        "folder": os.path.join(robocasa.__path__[0], "models/assets/textures"),
        "size_bytes": 538397915,
        "sha256": "d36bf23cec4ce33c83eb59dfd34faeb13d8f45f9844866a654569659fc9be23d",
    },
    "fixtures": {
        "message": "Downloading fixtures",
        "url": "https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip",
        "folder": os.path.join(robocasa.__path__[0], "models/assets/fixtures"),
        "size_bytes": 472441720,
        "sha256": "27905b494e7bf12826a8d67c167c73d95eade467841f5d2ddbdd87efc15b4c5a",
    },
    "objaverse": {
        "message": "Downloading Objaverse objects",
        "url": "https://utexas.box.com/shared/static/ejt1kc2v5vhae1rl4k5697i4xvpbjcox.zip",
        "folder": os.path.join(robocasa.__path__[0], "models/assets/objects/objaverse"),
        "size_bytes": 2115607481,
        "sha256": "519f989588c7e67f818ee823e1abeb7e08a34d9ec4977395a712dc8ce80c284c",
    },
    "generative_textures": {
        "message": "Downloading AI-generated environment textures",
        "url": "https://utexas.box.com/shared/static/gf9nkadvfrowkb9lmkcx58jwt4d6c1g3.zip",
        "folder": os.path.join(
            robocasa.__path__[0], "models/assets/generative_textures"
        ),
        "size_bytes": 593204309,
        "sha256": "a90d8ce74511120b19c859b326e3263e86be0a32eab10d6877e8a002d1b98c3d",
    },
}


class DownloadProgressBar(tqdm):
    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive(path, size_bytes, sha256):
    actual_size = Path(path).stat().st_size
    if actual_size != size_bytes:
        raise RuntimeError(
            f"size mismatch: expected {size_bytes} bytes, downloaded {actual_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != sha256:
        raise RuntimeError(
            f"SHA-256 mismatch: expected {sha256}, downloaded {actual_sha256}"
        )


def _safe_extract(archive_path, destination):
    destination = Path(destination).resolve()
    with ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if member_path != destination and destination not in member_path.parents:
                raise RuntimeError(f"unsafe archive path: {member.filename}")
        archive.extractall(destination)


def _install_extracted_tree(extraction_root, folder):
    extraction_root = Path(extraction_root)
    folder = Path(folder)
    extracted = extraction_root / folder.name
    if not extracted.is_dir():
        roots = list(extraction_root.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise RuntimeError(
                f"archive does not contain the expected {folder.name}/ directory"
            )
        extracted = roots[0]

    backup = folder.parent / f".{folder.name}.barx-backup"
    if backup.exists():
        raise RuntimeError(f"stale asset backup exists: {backup}")

    moved_existing = folder.exists()
    if moved_existing:
        folder.rename(backup)
    try:
        extracted.rename(folder)
    except Exception:
        if moved_existing:
            backup.rename(folder)
        raise
    if moved_existing:
        shutil.rmtree(backup)


def download_and_extract_zip(url, folder, size_bytes, sha256, message="Downloading..."):
    if not url.endswith(".zip"):
        raise ValueError(f"expected a .zip URL, got {url}")

    folder = Path(folder).resolve()
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive_path = folder.parent / f".{folder.name}.zip"
    partial_path = archive_path.with_suffix(".zip.part")
    print(colored(message, "yellow"))

    last_error = None
    for attempt in range(1, 4):
        try:
            partial_path.unlink(missing_ok=True)
            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc=folder.name
            ) as progress:
                urllib.request.urlretrieve(
                    url, filename=partial_path, reporthook=progress.update_to
                )
            _verify_archive(partial_path, size_bytes, sha256)
            partial_path.replace(archive_path)
            break
        except Exception as error:
            last_error = error
            partial_path.unlink(missing_ok=True)
            print(colored(f"Download attempt {attempt}/3 failed: {error}", "red"))
    else:
        raise RuntimeError(f"failed to download {url} after 3 attempts") from last_error

    print(colored("Checksum verified; extracting...", "yellow"))
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{folder.name}-", dir=folder.parent
        ) as extraction_root:
            _safe_extract(archive_path, extraction_root)
            _install_extracted_tree(extraction_root, folder)
    finally:
        archive_path.unlink(missing_ok=True)
    print(colored("Done.\n", "yellow"))


def download_kitchen_assets(asset_names=None, assume_yes=False):
    selected = list(DOWNLOAD_ASSET_REGISTRY) if asset_names is None else asset_names
    total_bytes = sum(DOWNLOAD_ASSET_REGISTRY[name]["size_bytes"] for name in selected)
    if not assume_yes:
        answer = input(
            f"Download and replace {total_bytes / 1024**3:.2f} GiB of kitchen assets? (y/n) "
        )
        if answer.lower() not in {"y", "yes"}:
            print("Aborting.")
            return

    for name in selected:
        download_and_extract_zip(**DOWNLOAD_ASSET_REGISTRY[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        action="append",
        choices=tuple(DOWNLOAD_ASSET_REGISTRY),
        dest="asset_names",
        help="download only this asset group (repeatable); default: all BARX-required groups",
    )
    parser.add_argument(
        "--yes", action="store_true", help="replace existing assets without prompting"
    )
    args = parser.parse_args()
    download_kitchen_assets(args.asset_names, assume_yes=args.yes)


if __name__ == "__main__":
    main()
