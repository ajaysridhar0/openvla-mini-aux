#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCKFILE="${ROOT_DIR}/repos.lock.json"
SRC_DIR="${ROOT_DIR}/src"

export PYTHONPATH="${SRC_DIR}/robosuite_xembod:${SRC_DIR}/robocasa_xembod:${SRC_DIR}/robomimic_xembod:${SRC_DIR}/mimicgen_xembod${PYTHONPATH:+:${PYTHONPATH}}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Could not find python3 or python in PATH." >&2
    exit 1
fi

"${PYTHON_BIN}" - "${LOCKFILE}" "${SRC_DIR}" <<'PY'
import importlib
import json
import sys
from pathlib import Path

lockfile = Path(sys.argv[1])
src_dir = Path(sys.argv[2])
entries = json.loads(lockfile.read_text(encoding="utf-8"))

errors = []

for entry in sorted(entries, key=lambda item: item["install_order"]):
    name = entry["name"]
    expected_commit = entry["commit"]
    repo_dir = src_dir / name
    git_dir = repo_dir / ".git"

    if not git_dir.exists():
        errors.append(f"missing clone: {repo_dir}")
        continue

    head = (
        __import__("subprocess")
        .check_output(["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    if head != expected_commit:
        errors.append(f"wrong commit for {name}: found {head}, expected {expected_commit}")

module_name = {
    "robosuite_xembod": "robosuite",
    "robocasa_xembod": "robocasa",
    "robomimic_xembod": "robomimic",
    "mimicgen_xembod": "mimicgen",
}

for repo_name, import_name in module_name.items():
    repo_dir = src_dir / repo_name
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # pragma: no cover - shell validation path
        errors.append(f"failed to import {import_name}: {exc}")
        continue

    module_origin = getattr(module, "__file__", None)
    if module_origin is None:
        module_path_entries = list(getattr(module, "__path__", []))
        if not module_path_entries:
            errors.append(f"could not determine import path for {import_name}")
            continue
        module_path = Path(module_path_entries[0]).resolve()
    else:
        module_path = Path(module_origin).resolve()

    try:
        module_path.relative_to(repo_dir.resolve())
    except ValueError:
        errors.append(
            f"{import_name} resolves to {module_path}, not the pinned clone under {repo_dir.resolve()}"
        )

robocasa_assets_root = src_dir / "robocasa_xembod" / "robocasa" / "models" / "assets"
required_asset_paths = [
    robocasa_assets_root / "fixtures" / "accessories" / "outlets" / "simple_white" / "visuals" / "outlet_0.obj",
    robocasa_assets_root / "textures",
    robocasa_assets_root / "generative_textures",
    robocasa_assets_root / "objects" / "objaverse",
]

for asset_path in required_asset_paths:
    if not asset_path.exists():
        errors.append(f"missing RoboCasa asset payload: {asset_path}")

for macros_path in [
    src_dir / "robosuite_xembod" / "robosuite" / "macros_private.py",
    src_dir / "robocasa_xembod" / "robocasa" / "macros_private.py",
]:
    if not macros_path.is_file():
        errors.append(f"missing generated private macros file: {macros_path}")

for extension_path in [
    src_dir / "mimicgen_xembod" / "mimicgen" / "configs" / "robocasa" / "single_stage" / "config_mug.py",
    src_dir / "mimicgen_xembod" / "mimicgen" / "env_interfaces" / "robocasa" / "single_stage" / "mg_mug.py",
]:
    if not extension_path.is_file():
        errors.append(f"missing BARX MimicGen extension: {extension_path}")

jaco_registry = src_dir / "mimicgen_xembod" / "mimicgen" / "utils" / "robomimic_utils.py"
if jaco_registry.is_file() and '"JacoOmron"' not in jaco_registry.read_text(encoding="utf-8"):
    errors.append("MimicGen robot-transfer registry is missing JacoOmron")

if errors:
    for err in errors:
        print(err, file=sys.stderr)
    sys.exit(1)

print("Pinned RoboCasa-X eval stack is present, locked, and importable.")
PY
