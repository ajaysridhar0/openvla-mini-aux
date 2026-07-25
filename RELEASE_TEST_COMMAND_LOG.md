# BARX public release exact acceptance commands

These are the verbatim command payloads preserved by the repository's `scripts/log_release_command.py`, with only external absolute paths replaced by the `$BARX_*`, `$HF_HOME`, and `$UV_CACHE_DIR` placeholders below.

The order, attempt number, exit code, duration, and stdout/stderr SHA-256 values come from the append-only command ledger. Failed attempts are intentionally retained.

```bash
export BARX_ACCEPTANCE_ROOT="/path/to/external/acceptance-run"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_COMMAND_LOG="$BARX_ACCEPTANCE_ROOT/command-log"
export BARX_COMMAND_SOURCE="$BARX_ACCEPTANCE_ROOT/bootstrap"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_RAW_XP900="$BARX_ACCEPTANCE_ROOT/raw-xp900-pnp"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export BARX_CONVERTED_ROOT="$BARX_ACCEPTANCE_ROOT/converted-rlds"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export BARX_RUNNER_BIN="$BARX_ACCEPTANCE_ROOT/runner-bin"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
```

The logger itself was acquired into the empty external run directory before the anonymous clone by fetching the exact-commit raw file over public HTTPS; the first substantive clone/preflight payload below was then recorded by that logger.

## `000-system-clone` attempt 1

- Section: 0
- Working directory: `/tmp`
- Result: FAIL (exit 1, 0.131 s)
- Stdout SHA-256: `ec2bb140e678b768e5221727dc94635bb23a8b638fc249adfec913579f120ff2`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_XP900="$BARX_ACCEPTANCE_ROOT/raw-xp900-pnp"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export BARX_CONVERTED_RLDS="$BARX_ACCEPTANCE_ROOT/converted-rlds"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
mkdir -p "$BARX_EVIDENCE"/{00-system,01-install,02-artifacts,03-hdf5,04-rlds,05-conversion,06-training,07-eval,08-mimicgen,09-final}
(
  date --iso-8601=seconds
  uname -a
  command -v git
  command -v cmake
  command -v nvidia-smi
  git --version
  cmake --version
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  df -h "$BARX_ACCEPTANCE_ROOT"
) 2>&1 | tee "$BARX_EVIDENCE/00-system/preflight.txt"
git clone --branch barx-release-integration \
  https://github.com/ajaysridhar0/openvla-mini-aux.git "$BARX_REPO"
cd "$BARX_REPO"
{
  git rev-parse HEAD
  git remote -v
  git status --porcelain
} | tee "$BARX_EVIDENCE/00-system/git.txt"
test "$(git rev-parse HEAD)" = "ce18937e1b11dd1b0d932fbe1d092b3433a8c28f"
```

## `000-system-clone` attempt 2

- Section: 0
- Working directory: `/tmp`
- Result: PASS (exit 0, 22.825 s)
- Stdout SHA-256: `bf1a60c0e217724e60a77a7a4991e67133d9d8be56027cef456f63fd0f82032f`
- Stderr SHA-256: `97d55ae830ae92a919a8201153b8ed10a6a0036beb1379c3125222def129d34c`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: explained runner-environment contamination
# CLEAN_STATE_NOTE: no checkout, BARX artifact cache, or Hugging Face cache existed after A01
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_XP900="$BARX_ACCEPTANCE_ROOT/raw-xp900-pnp"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export BARX_CONVERTED_RLDS="$BARX_ACCEPTANCE_ROOT/converted-rlds"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
test ! -e "$BARX_REPO"
test ! -e "$BARX_ARTIFACT_ROOT"
test ! -e "$HF_HOME"
mkdir -p "$BARX_EVIDENCE"/{00-system,01-install,02-artifacts,03-hdf5,04-rlds,05-conversion,06-training,07-eval,08-mimicgen,09-final}
(
  date --iso-8601=seconds
  uname -a
  command -v git
  command -v cmake
  command -v nvidia-smi
  git --version
  cmake --version
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  df -h "$BARX_ACCEPTANCE_ROOT"
) 2>&1 | tee "$BARX_EVIDENCE/00-system/preflight.txt"
git clone --branch barx-release-integration \
  https://github.com/ajaysridhar0/openvla-mini-aux.git "$BARX_REPO"
cd "$BARX_REPO"
{
  git rev-parse HEAD
  git remote -v
  git status --porcelain
} | tee "$BARX_EVIDENCE/00-system/git.txt"
test "$(git rev-parse HEAD)" = "ce18937e1b11dd1b0d932fbe1d092b3433a8c28f"
```

## `001-protocol-snapshot` attempt 1

- Section: 0
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.009 s)
- Stdout SHA-256: `b9f761aa9d377d9b3d786cb3d64ab50a6575b974cba38d013fe0a454e1b8e43e`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
cd "$BARX_REPO"
{
  sha256sum RELEASE_TEST_README.md scripts/log_release_command.py
  wc -l RELEASE_TEST_README.md scripts/log_release_command.py
} | tee "$BARX_EVIDENCE/00-system/protocol-snapshot.txt"
```

## `010-uv-version` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.012 s)
- Stdout SHA-256: `7cbbaba38fdf27577971eb7ac6bf416fdf4ceaa186947df48bb09b03a0a91203`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
# RUNNER_ENVIRONMENT_NOTE: system directories precede the contaminated user-local CMake launcher
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version | tee "$BARX_EVIDENCE/01-install/uv-version.txt"
```

## `011-uv-sync` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: FAIL (exit 130, 272.494 s)
- Stdout SHA-256: `0a4bc54514d1a71e970fe2bd39129fb2a5c6a9b58935557963b67abd6658c256`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv sync --locked --no-dev 2>&1 | tee "$BARX_EVIDENCE/01-install/uv-sync.txt"
```

## `011a-uv-sync-clean-state` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.009 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: explained runner shared-cache/filesystem hang
# CLEAN_STATE_NOTE: preserve and remove the partial A01 virtual environment before A02
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
mkdir -p "$BARX_ACCEPTANCE_ROOT/retry-quarantine"
if test -e "$BARX_REPO/.venv"; then
  mv "$BARX_REPO/.venv" "$BARX_ACCEPTANCE_ROOT/retry-quarantine/section1-A01-partial-venv"
fi
test ! -e "$BARX_REPO/.venv"
test ! -e "$BARX_ACCEPTANCE_ROOT/uv-cache"
```

## `011-uv-sync` attempt 2

- Section: 1
- Working directory: `$BARX_REPO`
- Result: FAIL (exit 1, 28.499 s)
- Stdout SHA-256: `89810f0eed9707c7baf4d7556450e7e42473206ab3158dc3afcf4efa0e5155fe`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: explained runner shared-cache/filesystem hang
# CLEAN_STATE_NOTE: no .venv exists; UV_CACHE_DIR is new and acceptance-local
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv sync --locked --no-dev 2>&1 | tee "$BARX_EVIDENCE/01-install/uv-sync.txt"
```

## `010-uv-version` attempt 2

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 2.019 s)
- Stdout SHA-256: `9e26c42212afa79f4cc83662aed6f52e1f7da3ff63ca8259a7f11bb3de412a7d`
- Stderr SHA-256: `b8be702ca036a9e49506ff78ed1b4618e1838fc534ae46717abd7b2820a19dd2`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: preinstalled uv 0.6.2 is below documented minimum 0.11.11
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version | tee "$BARX_EVIDENCE/01-install/uv-version.txt"
```

## `011b-uv-sync-clean-state` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.009 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: unmet documented uv prerequisite in A02
# CLEAN_STATE_NOTE: preserve A02 .venv and local uv cache, then recreate neither before A03
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
mkdir -p "$BARX_ACCEPTANCE_ROOT/retry-quarantine"
if test -e "$BARX_REPO/.venv"; then
  mv "$BARX_REPO/.venv" "$BARX_ACCEPTANCE_ROOT/retry-quarantine/section1-A02-partial-venv"
fi
if test -e "$BARX_ACCEPTANCE_ROOT/uv-cache"; then
  mv "$BARX_ACCEPTANCE_ROOT/uv-cache" "$BARX_ACCEPTANCE_ROOT/retry-quarantine/section1-A02-uv-cache"
fi
test ! -e "$BARX_REPO/.venv"
test ! -e "$BARX_ACCEPTANCE_ROOT/uv-cache"
```

## `011-uv-sync` attempt 3

- Section: 1
- Working directory: `$BARX_REPO`
- Result: FAIL (exit 1, 20.353 s)
- Stdout SHA-256: `56036de25ddfeffc50931c1d36ad26581034b8d3b63a12a7e6ccba8cc8ef1c06`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: documented uv prerequisite now satisfied
# CLEAN_STATE_NOTE: no .venv exists; UV_CACHE_DIR is new and acceptance-local
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv sync --locked --no-dev 2>&1 | tee "$BARX_EVIDENCE/01-install/uv-sync.txt"
```

## `011c-uv-sync-clean-state` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.014 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: known broken user-local CMake shadowed system CMake in A03
# CLEAN_STATE_NOTE: preserve A03 .venv and local uv cache, then expose only uv through a clean local bin
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
mkdir -p "$BARX_ACCEPTANCE_ROOT/retry-quarantine" "$BARX_ACCEPTANCE_ROOT/runner-bin"
if test -e "$BARX_REPO/.venv"; then
  mv "$BARX_REPO/.venv" "$BARX_ACCEPTANCE_ROOT/retry-quarantine/section1-A03-partial-venv"
fi
if test -e "$BARX_ACCEPTANCE_ROOT/uv-cache"; then
  mv "$BARX_ACCEPTANCE_ROOT/uv-cache" "$BARX_ACCEPTANCE_ROOT/retry-quarantine/section1-A03-uv-cache"
fi
ln -s "$HOME/.local/bin/uv" "$BARX_ACCEPTANCE_ROOT/runner-bin/uv"
test ! -e "$BARX_REPO/.venv"
test ! -e "$BARX_ACCEPTANCE_ROOT/uv-cache"
test -x "$BARX_ACCEPTANCE_ROOT/runner-bin/uv"
```

## `011-uv-sync` attempt 4

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 90.431 s)
- Stdout SHA-256: `2d655a4e02f0873e0bca918c3b180a45c64d740460c6e8a59dd840dc75148f94`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
# RETRY_CLASSIFICATION: known runner PATH contamination corrected without package changes
# CLEAN_STATE_NOTE: no .venv exists; UV_CACHE_DIR is new and acceptance-local
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv sync --locked --no-dev 2>&1 | tee "$BARX_EVIDENCE/01-install/uv-sync.txt"
```

## `012-unit-tests` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 2.820 s)
- Stdout SHA-256: `30b8abc22c401bd193d0f34df74b9c618e4f12e709215be72c936e40b424fcdf`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python -m unittest discover -s tests -v \
  2>&1 | tee "$BARX_EVIDENCE/01-install/unit-tests.txt"
```

## `013-assets` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 131.909 s)
- Stdout SHA-256: `970e0fa2aa84107a4656e3840c8d8aed50c48eea5a5ffa563e556d7b20284145`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python \
  robocasa_x/robocasa/scripts/download_kitchen_assets.py --yes \
  2>&1 | tee "$BARX_EVIDENCE/01-install/assets.txt"
```

## `014-simulator-import` attempt 1

- Section: 1
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 1.157 s)
- Stdout SHA-256: `b3bd7c3b02d5e9e677e5815a9a5eae3059dee12e45e717bbc10ce97d0a98ba16`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run --locked --no-dev python - 2>&1 <<'PY' | tee "$BARX_EVIDENCE/01-install/simulator-import.txt"
import mujoco
import robocasa
import robosuite

print("RoboCasa-X import OK")
print("MuJoCo", mujoco.__version__)
print("RoboCasa package", robocasa.__file__)
print("robosuite", robosuite.__version__)
PY
```

## `020-download-public-artifacts` attempt 1

- Section: 2
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 3999.193 s)
- Stdout SHA-256: `bbbbcb4bfbba0b03507a9eb3a491268fe7094e92ba8135f372c8db8475283d7b`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python scripts/download_public_artifacts.py \
  --artifact-root "$BARX_ARTIFACT_ROOT" \
  --include-pretrain-checkpoint \
  2>&1 | tee "$BARX_EVIDENCE/02-artifacts/download.txt"
```

## `021-checkpoint-collection` attempt 1

- Section: 2
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 1.441 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/02-artifacts/checkpoint-collection.json" <<'PY'
import json
import urllib.request

collection_slug = "ajaysri/barx-pretraining-models-joint-reps-and-no-reps"


def read_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": "barx-release-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


collection = read_json(f"https://huggingface.co/api/collections/{collection_slug}")
assert collection["private"] is False
assert len(collection["items"]) == 12, len(collection["items"])
models = []
for item in collection["items"]:
    assert item["type"] == "model"
    assert item["private"] is False
    assert item["gated"] is False
    model = read_json(f"https://huggingface.co/api/models/{item['id']}")
    checkpoints = [
        sibling["rfilename"]
        for sibling in model["siblings"]
        if sibling["rfilename"].endswith(".pt")
    ]
    assert len(model["sha"]) == 40
    assert len(checkpoints) == 1, (item["id"], checkpoints)
    models.append(
        {
            "repo_id": item["id"],
            "revision": model["sha"],
            "checkpoint": checkpoints[0],
            "private": model["private"],
            "gated": model["gated"],
        }
    )
print(json.dumps({"collection": collection["slug"], "models": models}, indent=2))
PY
```

## `022-local-inventory` attempt 1

- Section: 2
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.012 s)
- Stdout SHA-256: `bbd95e5b653ea53f1757fc1de021489c42175a3b6831781541b6c47ee90a3cca`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
cd "$BARX_REPO"
{
  find "$BARX_ARTIFACT_ROOT/base-vlm" -type f -printf '%P\t%s bytes\n' | sort
  find "$BARX_ARTIFACT_ROOT/vq" -type f -printf '%P\t%s bytes\n' | sort
  find "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps" \
    -type f -printf '%P\t%s bytes\n' | sort
} | tee "$BARX_EVIDENCE/02-artifacts/local-inventory.txt"
```

## `030-xp900-dry-run` attempt 1

- Section: 3
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.337 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_XP900="$BARX_ACCEPTANCE_ROOT/raw-xp900-pnp"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset xp_900 --task pnp \
  --output-dir "$BARX_RAW_XP900" --dry-run \
  > "$BARX_EVIDENCE/03-hdf5/xp900-dry-run.txt"
```

## `031-human-dry-run` attempt 1

- Section: 3
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.248 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset target_50 --target panda --task flip_mug \
  --output-dir "$BARX_RAW_HUMAN" --dry-run \
  > "$BARX_EVIDENCE/03-hdf5/human-dry-run.txt"
```

## `032-xp900-download` attempt 1

- Section: 3
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 1551.416 s)
- Stdout SHA-256: `91b063517f110c3be6fec47b6ab0c86c138bdb179ab3ff244495c94ba62e0ae4`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_XP900="$BARX_ACCEPTANCE_ROOT/raw-xp900-pnp"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset xp_900 --task pnp \
  --output-dir "$BARX_RAW_XP900" \
  2>&1 | tee "$BARX_EVIDENCE/03-hdf5/xp900-download-and-verify.txt"
```

## `033-human-download` attempt 1

- Section: 3
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 38.064 s)
- Stdout SHA-256: `3d4c56c39f66358660e2f6948cb8601bede6bf8d4c5eb248e777fde9ef75bdc2`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python scripts/download_raw_data.py \
  --dataset target_50 --target panda --task flip_mug \
  --output-dir "$BARX_RAW_HUMAN" \
  2>&1 | tee "$BARX_EVIDENCE/03-hdf5/human-download-and-verify.txt"
```

## `034-hdf5-inspection` attempt 1

- Section: 3
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.230 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_XP900="$BARX_ACCEPTANCE_ROOT/raw-xp900-pnp"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/03-hdf5/inspection.json" <<'PY'
import csv
import json
import os
from pathlib import Path

import h5py

roots = {
    "xp900_pnp": Path(os.environ["BARX_RAW_XP900"]),
    "target_panda_flip_mug": Path(os.environ["BARX_RAW_HUMAN"]),
}
report = {}
for label, root in roots.items():
    with (root / "subset-manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    sample_path = root / rows[0]["relative_path"]
    with h5py.File(sample_path, "r") as dataset:
        demo_name = sorted(dataset["data"].keys())[0]
        demo = dataset["data"][demo_name]
        ep_meta = json.loads(demo.attrs["ep_meta"])
        env_args = json.loads(dataset["data"].attrs["env_args"])
        report[label] = {
            "files": len(rows),
            "demonstrations": sum(int(row["demonstrations"]) for row in rows),
            "sample_relative_path": rows[0]["relative_path"],
            "sample_demo": demo_name,
            "sample_keys": sorted(demo.keys()),
            "action_shape": list(demo["actions"].shape),
            "agentview_shape": list(demo["obs"]["agentview_rgb"].shape),
            "environment": env_args["env_name"],
            "instruction": ep_meta["lang"],
        }
print(json.dumps(report, indent=2))
PY
```

## `040-published-rlds` attempt 1

- Section: 4
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.136 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/04-rlds/published-rlds.json" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BARX_ARTIFACT_ROOT"]) / "data" / "mg_pnp_lite"
info_paths = list(root.rglob("dataset_info.json"))
feature_paths = list(root.rglob("features.json"))
tfrecords = [
    path
    for path in root.rglob("*.tfrecord-*")
    if ".cache" not in path.parts
]
assert len(info_paths) == 1, info_paths
assert len(feature_paths) == 1, feature_paths
assert len(tfrecords) == 256, len(tfrecords)
info = json.loads(info_paths[0].read_text())
report = {
    "root": str(root),
    "dataset_info": str(info_paths[0].relative_to(root)),
    "features": str(feature_paths[0].relative_to(root)),
    "tfrecord_shards": len(tfrecords),
    "total_bytes": sum(path.stat().st_size for path in tfrecords),
    "splits": info.get("splits"),
}
print(json.dumps(report, indent=2))
PY
```

## `041-rlds-collection-audit` attempt 1

- Section: 4
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.283 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/04-rlds/collection-audit.json" <<'PY'
import json
import urllib.request
from pathlib import Path

from barx.raw_data import all_subsets

slug = "ajaysri/barx-rlds-datasets-69cad164926390ebbc395496"
request = urllib.request.Request(
    f"https://huggingface.co/api/collections/{slug}",
    headers={"User-Agent": "barx-release-audit"},
)
with urllib.request.urlopen(request, timeout=60) as response:
    collection = json.load(response)
actual = {item["id"] for item in collection["items"]}
expected = {subset.rlds_repo_id for subset in all_subsets()}
manifests = {path.stem for path in Path("dataset/subsets").glob("*.csv")}
assert collection["private"] is False
assert len(actual) == 24
assert actual == expected
assert manifests == {repo_id.split("/", 1)[1] for repo_id in expected}
assert all(not item["private"] and not item["gated"] for item in collection["items"])
print(
    json.dumps(
        {
            "collection": collection["slug"],
            "items": sorted(actual),
            "raw_subset_manifests": sorted(manifests),
        },
        indent=2,
    )
)
PY
```

## `050-conversion` attempt 1

- Section: 5
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 44.144 s)
- Stdout SHA-256: `34e8b053dd7ee823604b39c417e6cf93e13f4bc3f80b1de0d7c6f13beb472cc8`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export BARX_CONVERTED_RLDS="$BARX_ACCEPTANCE_ROOT/converted-rlds"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --no-dev python scripts/build_rlds.py \
  --dataset target_50 --target panda --task flip_mug \
  --raw-root "$BARX_RAW_HUMAN" \
  --rlds-root "$BARX_CONVERTED_RLDS" \
  2>&1 | tee "$BARX_EVIDENCE/05-conversion/conversion.txt"
```

## `051-converted-rlds` attempt 1

- Section: 5
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 4.945 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `b0991d75e66241e3525c38a1c145e18df97f05cc9e87a79782301d58271af218`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_CONVERTED_RLDS="$BARX_ACCEPTANCE_ROOT/converted-rlds"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/05-conversion/converted-rlds.json" <<'PY'
import json
import os
from pathlib import Path

import tensorflow_datasets as tfds

root = Path(os.environ["BARX_CONVERTED_RLDS"])
info_paths = list(root.rglob("dataset_info.json"))
assert len(info_paths) == 1, info_paths
dataset_dir = info_paths[0].parent
builder = tfds.builder_from_directory(str(dataset_dir))
dataset = builder.as_dataset(split="train", shuffle_files=False)
episode = next(iter(tfds.as_numpy(dataset.take(1))))
first = next(iter(episode["steps"]))
report = {
    "builder_name": builder.info.name,
    "version": str(builder.info.version),
    "examples": builder.info.splits["train"].num_examples,
    "step_keys": sorted(first),
    "observation_keys": sorted(first["observation"]),
    "action_shape": list(first["action"].shape),
    "instruction": first["language_instruction"].decode(),
}
assert report["examples"] == 50
assert report["action_shape"] == [7]
print(json.dumps(report, indent=2))
PY
```

## `060-prior-paper-dry-run` attempt 1

- Section: 6
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 11.826 s)
- Stdout SHA-256: `4457b36713f52064d4aaa1f693b6183cc0ed1e29763b8aaf834e19621378e707`
- Stderr SHA-256: `3fb1b1e36ab0c3ff9c925c6e31f27ac161636a1ae67eba2df2bbe370fd42913e`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/paper-dry-run" \
  --gpus 8 --global-batch-size 256 --per-device-batch-size 32 \
  --max-steps 50000 --dry-run \
  | tee "$BARX_EVIDENCE/06-training/prior-paper-command.txt"
```

## `061-adaptation-paper-dry-run` attempt 1

- Section: 6
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.120 s)
- Stdout SHA-256: `03b77cd1be8d485145f526ce2a5a2b379c4c630400b94da74702c8b7a87d7e1b`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv run --locked --extra train --no-dev python scripts/train.py adapt \
  --prior xp_900 --target panda --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/paper-dry-run" \
  --gpus 8 --global-batch-size 256 --per-device-batch-size 32 \
  --max-steps 3000 --dry-run \
  | tee "$BARX_EVIDENCE/06-training/adaptation-paper-command.txt"
```

## `062-train-sync` attempt 1

- Section: 6
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.064 s)
- Stdout SHA-256: `490e0e72e27d44de15282b5ca411cdb872760a8058c69d5510fd4ce19071f67d`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
uv sync --locked --extra train --no-dev \
  2>&1 | tee "$BARX_EVIDENCE/06-training/train-sync.txt"
```

## `063-one-step-training` attempt 1

- Section: 6
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 323.196 s)
- Stdout SHA-256: `78b3b454b1260ce79f15775fa539d874d5d99dba7fea0064e1a0c3d1ecda5269`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN WANDB_API_KEY
cd "$BARX_REPO"
uv run --locked --extra train --no-dev python scripts/train.py prior \
  --prior xp_900 --task pnp --method joint_reps \
  --data-root "$BARX_ARTIFACT_ROOT/data" \
  --base-vlm "$BARX_ARTIFACT_ROOT/base-vlm" \
  --run-root "$BARX_ARTIFACT_ROOT/runs/smoke" \
  --gpus 1 --global-batch-size 1 --per-device-batch-size 1 \
  --max-steps 1 --save-interval 100 --skip-final-checkpoint \
  2>&1 | tee "$BARX_EVIDENCE/06-training/one-step-training.txt"
```

## `064-training-artifacts` attempt 1

- Section: 6
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.160 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/06-training/training-artifacts.json" <<'PY'
import json
import math
import os
from pathlib import Path

run = (
    Path(os.environ["BARX_ARTIFACT_ROOT"])
    / "runs"
    / "smoke"
    / "joint_reps--xp_900_pnp"
)
required = ["config.json", "config.yaml", "dataset_statistics.json", "run-metrics.jsonl"]
missing = [name for name in required if not (run / name).is_file()]
assert not missing, missing
metric_files = sorted(run.glob("*.jsonl"))
records = []
for path in metric_files:
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if any(
            "loss" in key.lower() or key.lower().endswith("/step")
            for key in record
        ):
            records.append({"file": path.name, "record": record})
assert records, metric_files
for item in records:
    for value in item["record"].values():
        if isinstance(value, float):
            assert math.isfinite(value)
report = {
    "run": str(run),
    "required_files": required,
    "metric_files": [path.name for path in metric_files],
    "metric_records": records,
    "checkpoints_written": [
        path.name for path in (run / "checkpoints").glob("*.pt")
    ],
}
assert report["checkpoints_written"] == []
print(json.dumps(report, indent=2))
PY
```

## `070-eval-conditions` attempt 1

- Section: 7
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 2208.192 s)
- Stdout SHA-256: `38faaa4138585f9fb411678895fba3d101aa45b7989ccd28322e2c48f4cfbe9f`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN
cd "$BARX_REPO"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run --locked --no-dev python scripts/verify_eval_conditions.py \
  --task pnp_counter_to_sink --embodiment panda \
  2>&1 | tee "$BARX_EVIDENCE/07-eval/conditions.txt"
```

## `071-one-trial` attempt 1

- Section: 7
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 243.914 s)
- Stdout SHA-256: `aed4c0ea5b5a2bf8278912879a52756765876cbc6f6fa4077b39ea5da74df289`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export HF_HOME="$BARX_ACCEPTANCE_ROOT/hf"
export BARX_VQ_ROOT="$BARX_ARTIFACT_ROOT/vq"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN WANDB_API_KEY
cd "$BARX_REPO"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  uv run --locked --extra train --no-dev python scripts/evaluate.py \
  --checkpoint "$BARX_ARTIFACT_ROOT/runs/xp900-pnp-joint-reps/checkpoints/step-050000-epoch-15-loss=0.2577.pt" \
  --embodiment panda \
  --task pnp_counter_to_sink \
  --unnorm-key mg_pnp_lite \
  --episodes 1 \
  --rollout-dir "$BARX_ARTIFACT_ROOT/rollouts/one-trial" \
  2>&1 | tee "$BARX_EVIDENCE/07-eval/one-trial.txt"
```

## `072-summarize-evaluations` attempt 1

- Section: 7
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.165 s)
- Stdout SHA-256: `9636e7e823cd83618e3ceb53bdbaf9a1f3aba4762a8447cf2a4baafccb559cc3`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
uv run --locked --no-dev python scripts/summarize_evaluations.py \
  "$BARX_ARTIFACT_ROOT/rollouts" \
  --output "$BARX_EVIDENCE/07-eval/results.csv"
```

## `073-evaluation-artifacts` attempt 1

- Section: 7
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.100 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/07-eval/evaluation-artifacts.json" <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BARX_ARTIFACT_ROOT"]) / "rollouts"
summaries = list(root.rglob("summary.json"))
assert len(summaries) == 1, summaries
run = summaries[0].parent
summary = json.loads((run / "summary.json").read_text())
config = json.loads((run / "config.json").read_text())
episodes = [
    json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()
]
videos = sorted(path.name for path in run.glob("*.mp4"))
assert summary["status"] == "complete", summary
assert summary["episodes"] == 1, summary
assert len(episodes) == 1
assert episodes[0]["seed"] == 1000
assert videos
print(
    json.dumps(
        {
            "run_directory": str(run),
            "summary": summary,
            "episode": episodes[0],
            "checkpoint": config["pretrained_checkpoint"],
            "videos": videos,
        },
        indent=2,
    )
)
PY
```

## `074-evaluation-video-inspection` attempt 1

- Section: 7
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.341 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_ARTIFACT_ROOT="$BARX_ACCEPTANCE_ROOT/artifacts"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
video="$(find "$BARX_ARTIFACT_ROOT/rollouts" -type f -name '*.mp4' -print -quit)"
test -n "$video"
ffprobe -v error -show_format -show_streams -of json "$video" \
  > "$BARX_EVIDENCE/07-eval/video-metadata.json"
ffmpeg -hide_banner -loglevel error -y -i "$video" \
  -vf "fps=1/5,scale=320:-1,tile=4x3" -frames:v 1 \
  "$BARX_EVIDENCE/07-eval/video-contact-sheet.png"
sha256sum "$video" "$BARX_EVIDENCE/07-eval/video-contact-sheet.png" \
  > "$BARX_EVIDENCE/07-eval/video-sha256.txt"
```

## `080-mimicgen-sync` attempt 1

- Section: 8
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 1.210 s)
- Stdout SHA-256: `3b153ab2499189f0926dc7074776116eec7b60cc08d1e50c1a9ddced67110055`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
uv sync --locked --extra mg --no-dev \
  2>&1 | tee "$BARX_EVIDENCE/08-mimicgen/mg-sync.txt"
```

## `081-mimicgen-interface` attempt 1

- Section: 8
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.486 s)
- Stdout SHA-256: `f2bdeffd54634d77f8c19b6afd49b197289b6a52f22ea5c337f80ac0deb78251`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
{
  test -f third_party/mimicgen/LICENSE
  uv run --locked --extra mg --no-dev python \
    scripts/prepare_mimicgen_source.py --help
  uv run --locked --extra mg --no-dev python \
    scripts/generate_mimicgen.py --help
} 2>&1 | tee "$BARX_EVIDENCE/08-mimicgen/interface.txt"
```

## `082-mimicgen-prepare` attempt 1

- Section: 8
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 81.109 s)
- Stdout SHA-256: `f3bfddefb3ad712a3328197bf814fa635c5d53827286e2cb5d3e737123a94679`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_RAW_HUMAN="$BARX_ACCEPTANCE_ROOT/raw-target-panda-flip-mug"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export BARX_MG_SOURCE="$BARX_RAW_HUMAN/human/PandaOmron/FlipMugUpright/demo_gentex_im320.hdf5"
export BARX_MG_PREPARED="$BARX_MG_ROOT/panda-flip-mug-prepared.hdf5"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
mkdir -p "$BARX_MG_ROOT"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run --locked --extra mg --no-dev python \
  scripts/prepare_mimicgen_source.py \
  --source "$BARX_MG_SOURCE" \
  --output "$BARX_MG_PREPARED" \
  --task flip_mug_upright \
  --demos 5 \
  --summary "$BARX_EVIDENCE/08-mimicgen/preparation-summary.json" \
  2>&1 | tee "$BARX_EVIDENCE/08-mimicgen/preparation.txt"
```

## `083-mimicgen-generate` attempt 1

- Section: 8
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 49.660 s)
- Stdout SHA-256: `daf3db602a2cfbb4f30b910d38b083b1e01077debbdef9592b0c25b5d9ff07b4`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export BARX_MG_PREPARED="$BARX_MG_ROOT/panda-flip-mug-prepared.hdf5"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
mkdir -p "$BARX_MG_ROOT/gl-cache" "$BARX_MG_ROOT/xdg-cache"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
__GL_SHADER_DISK_CACHE_PATH="$BARX_MG_ROOT/gl-cache" \
XDG_CACHE_HOME="$BARX_MG_ROOT/xdg-cache" \
uv run --locked --extra mg --no-dev python scripts/generate_mimicgen.py \
  --source "$BARX_MG_PREPARED" \
  --task flip_mug_upright \
  --embodiment panda \
  --seed 0 \
  --successes 1 \
  --max-attempts 25 \
  --source-demos 5 \
  --output-dir "$BARX_MG_ROOT/panda-flip-mug-seed0" \
  --video "$BARX_MG_ROOT/panda-flip-mug-seed0.mp4" \
  2>&1 | tee "$BARX_EVIDENCE/08-mimicgen/generation.txt"
```

## `084-mimicgen-artifacts` attempt 1

- Section: 8
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.111 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export PATH="$BARX_ACCEPTANCE_ROOT/runner-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export UV_CACHE_DIR="$BARX_ACCEPTANCE_ROOT/uv-cache"
cd "$BARX_REPO"
uv run --locked --no-dev python - \
  > "$BARX_EVIDENCE/08-mimicgen/artifacts.json" <<'PY'
import json
import os
from pathlib import Path

evidence = Path(os.environ["BARX_EVIDENCE"]) / "08-mimicgen"
root = Path(os.environ["BARX_MG_ROOT"])
preparation = json.loads((evidence / "preparation-summary.json").read_text())
generation = json.loads(
    (root / "panda-flip-mug-seed0" / "generation-summary.json").read_text()
)
config = json.loads(
    (root / "panda-flip-mug-seed0" / "resolved-config.json").read_text()
)

assert preparation["source_unchanged"] is True
assert preparation["source_sha256_before"] == preparation["source_sha256_after"]
assert preparation["output"]["prepared_demonstrations"] == 5
assert preparation["output"]["action_widths"] == [12]
assert generation["requested_successes"] == 1
assert generation["stats"]["num_success"] == 1
assert 1 <= generation["stats"]["num_attempts"] <= 25
assert generation["generated_hdf5"]
assert all(item["action_widths"] == [12] for item in generation["generated_hdf5"])
assert all(
    item["action_layout"] == "arm_gripper_base_torso_mode_v1"
    for item in generation["generated_hdf5"]
)
video = root / "panda-flip-mug-seed0.mp4"
assert video.is_file() and video.stat().st_size > 0

print(
    json.dumps(
        {
            "source_unchanged": preparation["source_unchanged"],
            "prepared_demonstrations": 5,
            "resolved_task": config["name"],
            "resolved_robot": config["experiment"]["task"]["robot"],
            "seed": generation["seed"],
            "stats": generation["stats"],
            "generated_hdf5": generation["generated_hdf5"],
            "video": {"name": video.name, "bytes": video.stat().st_size},
        },
        indent=2,
    )
)
PY
```

## `085-mimicgen-video-inspection` attempt 1

- Section: 8
- Working directory: `$BARX_REPO`
- Result: FAIL (exit 1, 0.267 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `7e000d8d2e2fd3ffd118f6425309cf74eb9e23102281f7f5aea06614c0349033`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
video="$BARX_MG_ROOT/panda-flip-mug-seed0.mp4"
test -s "$video"
ffprobe -v error -show_format -show_streams -of json "$video" \
  > "$BARX_EVIDENCE/08-mimicgen/video-metadata.json"
ffmpeg -hide_banner -loglevel error -y -i "$video" \
  -vf "fps=1/5,scale=320:-1,tile=4x3" -frames:v 1 \
  "$BARX_EVIDENCE/08-mimicgen/video-contact-sheet.png"
sha256sum "$video" "$BARX_EVIDENCE/08-mimicgen/video-contact-sheet.png" \
  > "$BARX_EVIDENCE/08-mimicgen/video-sha256.txt"
```

## `085-mimicgen-video-inspection` attempt 2

- Section: 8
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.391 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
video="$BARX_MG_ROOT/panda-flip-mug-seed0.mp4"
test -s "$video"
test -z "$(git -C "$BARX_REPO" status --porcelain)"
printf '%s\n' \
  "Attempt 1 classification: explained auxiliary review-harness sampling failure; the 2.35-second video yields no frame at fps=1/5. Repository clean and original generated video retained." \
  > "$BARX_EVIDENCE/08-mimicgen/video-inspection-retry.txt"
ffprobe -v error -show_format -show_streams -of json "$video" \
  > "$BARX_EVIDENCE/08-mimicgen/video-metadata.json"
ffmpeg -hide_banner -loglevel error -y -i "$video" \
  -vf "fps=5,scale=320:-1,tile=4x3" -frames:v 1 \
  "$BARX_EVIDENCE/08-mimicgen/video-contact-sheet.png"
sha256sum "$video" "$BARX_EVIDENCE/08-mimicgen/video-contact-sheet.png" \
  > "$BARX_EVIDENCE/08-mimicgen/video-sha256.txt"
```

## `086-mimicgen-endpoints` attempt 1

- Section: 8
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.219 s)
- Stdout SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_MG_ROOT="$BARX_ACCEPTANCE_ROOT/mimicgen"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
video="$BARX_MG_ROOT/panda-flip-mug-seed0.mp4"
test -s "$video"
ffmpeg -hide_banner -loglevel error -y -i "$video" \
  -vf "select='eq(n,0)+eq(n,46)',scale=512:-1,tile=2x1" -frames:v 1 \
  "$BARX_EVIDENCE/08-mimicgen/video-endpoints.png"
sha256sum "$BARX_EVIDENCE/08-mimicgen/video-endpoints.png" \
  > "$BARX_EVIDENCE/08-mimicgen/video-endpoints-sha256.txt"
```

## `100-git-clean` attempt 1

- Section: 10
- Working directory: `$BARX_REPO`
- Result: PASS (exit 0, 0.029 s)
- Stdout SHA-256: `f4631b2288db0aafddabe5123c7d7df02b696424395b9448e68a95ad805af713`
- Stderr SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`

```bash
#!/usr/bin/env bash
set -euo pipefail
export BARX_ACCEPTANCE_ROOT="$BARX_ACCEPTANCE_ROOT"
export BARX_REPO="$BARX_ACCEPTANCE_ROOT/repo"
export BARX_EVIDENCE="$BARX_ACCEPTANCE_ROOT/evidence"
cd "$BARX_REPO"
{
  git status --porcelain
  git diff --exit-code
  git diff --cached --exit-code
  git rev-parse HEAD
} 2>&1 | tee "$BARX_EVIDENCE/09-final/git-clean.txt"
```

## Corroborating no-context MimicGen runner

A second runner used a separate fresh clone, cache, and append-only evidence
root at the same exact commit. Its 21 command records had these outcomes:

| Command | Attempt(s) | Result |
| --- | ---: | --- |
| Anonymous clone | 1 | PASS |
| System/README capture | 3 | PASS |
| Preflight | 2 | FAIL, then PASS with system CMake |
| Locked `mg` sync | 1 | PASS |
| Full unit suite | 1 | PASS — 72 tests |
| Human HDF5 dry run/download | 2 | PASS |
| Public interface/license | 1 | PASS |
| Source preparation | 2 | FAIL before assets, then PASS after the documented asset step |
| Public asset install | 1 | PASS |
| Bounded generation | 1 | PASS — one success on attempt 1/25 |
| Artifact/HDF5 assertions | 1 | PASS |
| Video metadata/contact sheet | 2 | PASS |
| Git cleanliness | 1 | PASS |
| Final inventory helper | 2 | FAIL on a report-only path assumption, then PASS |

The core release commands and arguments were identical to command IDs
`080`–`084` and `100` above. This independent run produced a 246-step
canonical HDF5 and a 50-frame H.264 video; the source SHA-256 stayed unchanged,
the visible mug flip succeeded, and Git remained clean.
