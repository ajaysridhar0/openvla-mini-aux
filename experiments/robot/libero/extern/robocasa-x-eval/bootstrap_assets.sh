#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${ROOT_DIR}/src"
ROBOCASA_REPO_DIR="${SRC_DIR}/robocasa_xembod"
ROBOCASA_ASSETS_DIR="${ROBOCASA_REPO_DIR}/robocasa/models/assets"
ASSET_MIRROR_ROOT="${ROBOCASA_EVAL_ASSETS_MIRROR_ROOT:-}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Could not find python3 or python in PATH." >&2
    exit 1
fi

if [[ ! -d "${ROBOCASA_REPO_DIR}/.git" ]]; then
    echo "Missing robocasa_xembod clone at ${ROBOCASA_REPO_DIR}." >&2
    echo "Run: ${ROOT_DIR}/bootstrap.sh" >&2
    exit 1
fi

mkdir -p "${ROBOCASA_ASSETS_DIR}"

sync_from_mirror() {
    local asset_name="$1"
    local source_path="${ASSET_MIRROR_ROOT}/${asset_name}"
    local dest_path="${ROBOCASA_ASSETS_DIR}/${asset_name}"

    if [[ -z "${ASSET_MIRROR_ROOT}" || ! -e "${source_path}" ]]; then
        return 0
    fi

    if [[ ! -e "${dest_path}" ]]; then
        echo "Linking RoboCasa asset ${asset_name} from ${source_path}"
        ln -s "${source_path}" "${dest_path}"
        return 0
    fi

    if [[ -L "${dest_path}" ]]; then
        return 0
    fi

    if [[ -d "${source_path}" && -d "${dest_path}" ]]; then
        echo "Syncing missing RoboCasa asset files for ${asset_name} from ${source_path}"
        cp -rn "${source_path}/." "${dest_path}/"
    fi
}

sync_from_mirror "fixtures"
sync_from_mirror "objects"
sync_from_mirror "textures"
sync_from_mirror "generative_textures"

missing_assets=()
[[ -f "${ROBOCASA_ASSETS_DIR}/fixtures/accessories/outlets/simple_white/visuals/outlet_0.obj" ]] || missing_assets+=("fixtures")
[[ -d "${ROBOCASA_ASSETS_DIR}/objects/objaverse" ]] || missing_assets+=("objaverse")
[[ -d "${ROBOCASA_ASSETS_DIR}/textures" ]] || missing_assets+=("textures")
[[ -d "${ROBOCASA_ASSETS_DIR}/generative_textures" ]] || missing_assets+=("generative_textures")

if [[ "${#missing_assets[@]}" -eq 0 ]]; then
    echo "RoboCasa kitchen assets already present under ${ROBOCASA_ASSETS_DIR}"
    exit 0
fi

echo "Downloading missing RoboCasa kitchen assets: ${missing_assets[*]}"

"${PYTHON_BIN}" - "${ROBOCASA_REPO_DIR}" "${missing_assets[@]}" <<'PY'
import sys
from pathlib import Path

robocasa_repo_dir = Path(sys.argv[1]).resolve()
assets_to_fetch = sys.argv[2:]

sys.path.insert(0, str(robocasa_repo_dir))

from robocasa.scripts.download_kitchen_assets import (  # noqa: E402
    DOWNLOAD_ASSET_REGISTRY,
    download_and_extract_zip,
)

asset_config_keys = {
    "fixtures": "fixtures",
    "objaverse": "objaverse",
    "textures": "textures",
    "generative_textures": "generative_textures",
}

for asset_name in assets_to_fetch:
    config = dict(DOWNLOAD_ASSET_REGISTRY[asset_config_keys[asset_name]])
    download_and_extract_zip(prompt_before_download=False, **config)
PY

echo "RoboCasa kitchen assets are ready under ${ROBOCASA_ASSETS_DIR}"
