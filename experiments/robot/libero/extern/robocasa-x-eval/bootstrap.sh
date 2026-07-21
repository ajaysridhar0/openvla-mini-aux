#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCKFILE="${ROOT_DIR}/repos.lock.json"
SRC_DIR="${ROOT_DIR}/src"
MIMICGEN_PATCH="${ROOT_DIR}/mimicgen_flip_mug.patch"
ROBOMIMIC_PATCH="${ROOT_DIR}/robomimic_lazy_imports.patch"
ROBOCASA_PATCH="${ROOT_DIR}/robocasa_lazy_imports.patch"
MIRROR_ROOT="${ROBOCASA_EVAL_DEPS_MIRROR_ROOT:-}"

mkdir -p "${SRC_DIR}"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "Could not find python3 or python in PATH." >&2
    exit 1
fi

mapfile -t LOCK_ROWS < <(
    "${PYTHON_BIN}" - "${LOCKFILE}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as infile:
    entries = json.load(infile)

for entry in sorted(entries, key=lambda item: item["install_order"]):
    print("\t".join([entry["name"], entry["url"], entry["commit"], str(entry["install_order"])]))
PY
)

unmanaged_changes() {
    local name="$1"
    local repo_dir="$2"
    local status

    status="$(git -C "${repo_dir}" status --porcelain --untracked-files=all)"
    case "${name}" in
        robocasa_xembod)
            printf '%s\n' "${status}" | grep -Ev '^.. robocasa/(models/assets/(fixtures|objects|textures|generative_textures)(/|$)|utils/robomimic/robomimic_dataset_utils\.py$)' || true
            ;;
        robosuite_xembod)
            printf '%s\n' "${status}" | grep -Ev '^.. robosuite/macros_private.py$' || true
            ;;
        mimicgen_xembod)
            printf '%s\n' "${status}" | grep -Ev \
                '^.. mimicgen/(__init__\.py|configs/__init__\.py|configs/robocasa/single_stage/config_mug\.py|env_interfaces/robocasa/single_stage/mg_mug\.py|utils/robomimic_utils\.py)$' || true
            ;;
        robomimic_xembod)
            printf '%s\n' "${status}" | grep -Ev '^.. robomimic/utils/file_utils\.py$' || true
            ;;
        *)
            printf '%s\n' "${status}"
            ;;
    esac
}

for row in "${LOCK_ROWS[@]}"; do
    IFS=$'\t' read -r name url commit _ <<<"${row}"
    repo_dir="${SRC_DIR}/${name}"
    clone_source="${url}"
    mirror_repo_dir=""

    mirror_name="${name}"
    if [[ "${name}" == "mimicgen_xembod" ]]; then
        mirror_name="mg_robocasa_xembod"
    fi
    if [[ -n "${MIRROR_ROOT}" && -d "${MIRROR_ROOT}/${mirror_name}/.git" ]]; then
        mirror_repo_dir="${MIRROR_ROOT}/${mirror_name}"
    fi

    if [[ -d "${repo_dir}" && ! -d "${repo_dir}/.git" ]]; then
        echo "Refusing to reuse non-git directory: ${repo_dir}" >&2
        exit 1
    fi

    if [[ ! -d "${repo_dir}/.git" ]]; then
        if [[ -n "${mirror_repo_dir}" ]]; then
            echo "Seeding ${name} from local mirror metadata at ${mirror_repo_dir}"
            mkdir -p "${repo_dir}"
            cp -R "${mirror_repo_dir}/.git" "${repo_dir}/.git"
        else
            echo "Cloning ${name} from ${clone_source} into ${repo_dir}"
            git clone "${clone_source}" "${repo_dir}"
        fi
    else
        if [[ -n "$(unmanaged_changes "${name}" "${repo_dir}")" ]]; then
            echo "Refusing to switch ${name}: working tree is dirty at ${repo_dir}" >&2
            exit 1
        fi
        if [[ -n "${mirror_repo_dir}" ]]; then
            echo "Using existing mirror-backed clone for ${name}; skipping fetch"
        else
            echo "Fetching updates for ${name}"
            git -C "${repo_dir}" fetch --tags --prune origin
        fi
    fi

    echo "Checking out ${name} at ${commit}"
    git -C "${repo_dir}" checkout --detach "${commit}"
    git -C "${repo_dir}" reset --hard "${commit}"

    if [[ "${name}" == "mimicgen_xembod" ]]; then
        echo "Applying the small BARX FlipMugUpright extension"
        git -C "${repo_dir}" clean -f -- \
            mimicgen/configs/robocasa/single_stage/config_mug.py \
            mimicgen/env_interfaces/robocasa/single_stage/mg_mug.py
        git -C "${repo_dir}" apply "${MIMICGEN_PATCH}"
    fi

    if [[ "${name}" == "robomimic_xembod" ]]; then
        echo "Applying lazy policy imports for data-generation startup"
        git -C "${repo_dir}" apply "${ROBOMIMIC_PATCH}"
    fi

    if [[ "${name}" == "robocasa_xembod" ]]; then
        echo "Applying lazy action-conversion imports for state rendering"
        git -C "${repo_dir}" apply "${ROBOCASA_PATCH}"
    fi

    echo "Using ${name} directly from the pinned source tree"
done

export PYTHONPATH="${SRC_DIR}/robosuite_xembod:${SRC_DIR}/robocasa_xembod:${SRC_DIR}/robomimic_xembod:${SRC_DIR}/mimicgen_xembod${PYTHONPATH:+:${PYTHONPATH}}"

ROBOSUITE_MACROS="${SRC_DIR}/robosuite_xembod/robosuite/macros_private.py"
if [[ ! -f "${ROBOSUITE_MACROS}" ]]; then
    echo "Creating the RoboSuite private macro file"
    cp "${SRC_DIR}/robosuite_xembod/robosuite/macros.py" "${ROBOSUITE_MACROS}"
fi

ROBOCASA_MACROS="${SRC_DIR}/robocasa_xembod/robocasa/macros_private.py"
if [[ ! -f "${ROBOCASA_MACROS}" ]]; then
    echo "Creating the RoboCasa private macro file"
    cp "${SRC_DIR}/robocasa_xembod/robocasa/macros.py" "${ROBOCASA_MACROS}"
fi

"${ROOT_DIR}/bootstrap_assets.sh"

echo
echo "Pinned RoboCasa-X eval stack is installed under ${SRC_DIR}"
