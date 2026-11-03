#!/usr/bin/env bash
# Install external SDKs; release files and model assets remain unchanged.
set -euo pipefail

RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON=python3.11
EXISTING_SIM_PYTHON=""
ENV_ROOT="$RELEASE_ROOT/.venv"
ISAACLAB_DIR="$RELEASE_ROOT/third_party/IsaacLab"
ISAACLAB_COMMIT=f4aa17f87e2e5db5484f0b5974918573e8918ce2
DRY_RUN=0
IK_ONLY=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [options]

  --python PATH          Python 3.11 executable for a new simulation environment.
  --sim-python PATH      Reuse an existing compatible Isaac Sim / IsaacLab Python;
                         no packages are installed into that environment.
  --env-root DIR         Environment location (default: release/.venv).
  --isaaclab-dir DIR     SDK checkout location (default: release/third_party/IsaacLab).
  --ik-only              Create/verify only the isolated Pink IK environment
                         (<env-root>/ik) and record NEPES_IK_PYTHON in env.sh.
  --dry-run              Print the installation plan; do not create or install files.
  -h, --help             Show this help.

New installation: Isaac Sim 5.1.0, pinned IsaacLab 2.3.2, PyTorch CUDA 12.8.
Requires Linux x86_64, Python 3.11, Git, internet, and a supported NVIDIA GPU
with RT cores and a working Vulkan driver for RGB cameras.
The NVIDIA EULA is not automatically accepted by this script.
An isolated dataset environment shares simulator libraries via a .pth file,
while installing its own LeRobot exporter dependencies.
Pink IK (default upper-body IK backend): a separate Python 3.11 venv
<env-root>/ik, created from --sim-python (or --python) without system
site-packages, receives exactly requirements/ik_pink.txt. Nothing is installed
into the simulator Python. env.sh exports NEPES_IK_PYTHON for the runtime.
After setup, source <env-root>/env.sh before running the collection launcher.
EOF
}

while (($#)); do
    case "$1" in
        --python|--sim-python|--env-root|--isaaclab-dir)
            [[ $# -ge 2 && -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
            case "$1" in
                --python) BASE_PYTHON="$2" ;;
                --sim-python) EXISTING_SIM_PYTHON="$2" ;;
                --env-root) ENV_ROOT="$2" ;;
                --isaaclab-dir) ISAACLAB_DIR="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --ik-only) IK_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Absolute paths are necessary for activation and the dataset fallback .pth.
ENV_ROOT="$(realpath -m -- "$ENV_ROOT")"
ISAACLAB_DIR="$(realpath -m -- "$ISAACLAB_DIR")"
SIM_PYTHON="${EXISTING_SIM_PYTHON:-$ENV_ROOT/sim/bin/python}"
DATASET_PYTHON="$ENV_ROOT/dataset/bin/python"
IK_PYTHON="$ENV_ROOT/ik/bin/python"

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if (( ! DRY_RUN )); then "$@"; fi
}

check_python() {
    "$1" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
}

setup_ik_env() {
    # Pink IK worker venv: a Python 3.11 base without system site-packages; only
    # the pinned closure is installed, so the simulator and dataset Pythons stay untouched.
    local base="$1"
    if (( ! DRY_RUN )); then
        check_python "$base"
        if [[ -e "$ENV_ROOT/ik" && ! -x "$IK_PYTHON" ]]; then
            echo 'Existing IK environment is incomplete; remove it or choose another --env-root.' >&2
            exit 2
        fi
    fi
    run "$base" -m venv "$ENV_ROOT/ik"
    run "$IK_PYTHON" -m pip install --disable-pip-version-check --no-input --no-deps --only-binary=:all: \
        -r "$RELEASE_ROOT/requirements/ik_pink.txt"
    run "$IK_PYTHON" -m pip check
    # Exact pins, imports, worker source hashes and a Pink QP smoke test (no simulator), with the
    # native library environment removed exactly as the runtime starts the worker (-I covers PYTHON*).
    run env -u LD_LIBRARY_PATH -u LD_PRELOAD "$IK_PYTHON" -I -B "$RELEASE_ROOT/ik/pink/worker.py" --check
}

if (( IK_ONLY )); then
    IK_BASE_PYTHON="${EXISTING_SIM_PYTHON:-$BASE_PYTHON}"
    if (( ! DRY_RUN )); then
        command -v -- "$IK_BASE_PYTHON" >/dev/null || { echo "Python not found: $IK_BASE_PYTHON" >&2; exit 2; }
        IK_BASE_PYTHON="$(realpath -ms -- "$(command -v -- "$IK_BASE_PYTHON")")"
        mkdir -p -- "$ENV_ROOT"
    fi
    setup_ik_env "$IK_BASE_PYTHON"
    if (( DRY_RUN )); then
        printf '+ record NEPES_IK_PYTHON=%q in %s/env.sh (other lines kept)\n' "$IK_PYTHON" "$ENV_ROOT"
        exit 0
    fi
    ENV_TMP="$(mktemp "$ENV_ROOT/.env.sh.XXXXXX")"
    if [[ -f "$ENV_ROOT/env.sh" ]]; then
        grep -v '^export NEPES_IK_PYTHON=' "$ENV_ROOT/env.sh" > "$ENV_TMP" || true
    fi
    printf 'export NEPES_IK_PYTHON=%q\n' "$IK_PYTHON" >> "$ENV_TMP"
    chmod "$(printf '%04o' "$(( 0666 & ~$(umask) ))")" -- "$ENV_TMP"  # same mode as a full setup's env.sh
    mv -- "$ENV_TMP" "$ENV_ROOT/env.sh"
    printf '\nPink IK environment ready. Activate with:\n  source %q\n' "$ENV_ROOT/env.sh"
    exit 0
fi

if [[ -z "$EXISTING_SIM_PYTHON" ]]; then
    if (( ! DRY_RUN )); then
        check_python "$BASE_PYTHON"
        command -v git >/dev/null || { echo 'Git is required.' >&2; exit 2; }
        if [[ -e "$ENV_ROOT/sim" && ! -x "$SIM_PYTHON" ]]; then
            echo 'Existing simulation environment is incomplete; choose another --env-root.' >&2
            exit 2
        fi
    fi
    IK_BASE_PYTHON="$BASE_PYTHON"
    run "$BASE_PYTHON" -m venv "$ENV_ROOT/sim"
    run "$SIM_PYTHON" -m pip install --upgrade pip
    run "$SIM_PYTHON" -m pip install 'torch==2.7.0' 'torchvision==0.22.0' 'torchaudio==2.7.0' \
        --index-url https://download.pytorch.org/whl/cu128
    run "$SIM_PYTHON" -m pip install 'isaacsim[all,extscache]==5.1.0' \
        --extra-index-url https://pypi.nvidia.com \
        -c "$RELEASE_ROOT/requirements/sim-constraints.txt"
    if (( DRY_RUN )) || [[ ! -e "$ISAACLAB_DIR" ]]; then
        run mkdir -p -- "$(dirname -- "$ISAACLAB_DIR")"
        run git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
        run git -C "$ISAACLAB_DIR" checkout --detach "$ISAACLAB_COMMIT"
    else
        [[ -d "$ISAACLAB_DIR/.git" ]] || { echo 'IsaacLab directory is not a Git checkout.' >&2; exit 2; }
        [[ "$(git -C "$ISAACLAB_DIR" rev-parse HEAD)" == "$ISAACLAB_COMMIT" ]] || {
            echo 'Existing IsaacLab checkout differs; select an empty --isaaclab-dir.' >&2; exit 2;
        }
        [[ -z "$(git -C "$ISAACLAB_DIR" status --porcelain)" ]] || {
            echo 'Existing IsaacLab checkout has local changes; select an empty --isaaclab-dir.' >&2; exit 2;
        }
    fi
    run "$SIM_PYTHON" -m pip install -c "$RELEASE_ROOT/requirements/sim-constraints.txt" \
        -r "$RELEASE_ROOT/requirements/runtime.txt"
    # Collection uses the SDK APIs, not its training/teleoperation extras.
    run "$SIM_PYTHON" -m pip install --no-deps \
        -e "$ISAACLAB_DIR/source/isaaclab" \
        -e "$ISAACLAB_DIR/source/isaaclab_contrib" \
        -e "$ISAACLAB_DIR/source/isaaclab_assets" \
        -e "$ISAACLAB_DIR/source/isaaclab_tasks"
else
    if (( ! DRY_RUN )); then
        command -v -- "$EXISTING_SIM_PYTHON" >/dev/null || { echo "Python not found: $EXISTING_SIM_PYTHON" >&2; exit 2; }
        SIM_PYTHON="$(command -v -- "$EXISTING_SIM_PYTHON")"
        # Preserve venv/bin/python symlinks: resolving them would bypass the venv.
        SIM_PYTHON="$(realpath -ms -- "$SIM_PYTHON")"
        check_python "$SIM_PYTHON"
    fi
    IK_BASE_PYTHON="$SIM_PYTHON"
    printf 'Reusing simulation environment without modifying it: %s\n' "$SIM_PYTHON"
fi

if (( DRY_RUN )); then
    printf '%s\n' '+ verify simulator package versions and required imports (no simulator launch)'
else
    "$SIM_PYTHON" - <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
expected = {
    "isaacsim": "5.1.0.0", "isaaclab": "0.54.3", "isaaclab_contrib": "0.0.2", "isaaclab_assets": "0.2.4",
    "isaaclab_tasks": "0.11.14", "numpy": "1.26.0", "scipy": "1.15.3",
    "torch": "2.7.0", "torchvision": "0.22.0", "gymnasium": "1.2.1",
}
for name, wanted in expected.items():
    actual = version(name).split("+")[0]
    if actual != wanted:
        raise SystemExit(f"{name}: expected {wanted}, found {actual}")
import numpy, scipy, torch, torchvision, gymnasium, PIL, trimesh, rtree, yaml, toml, prettytable
import hid, h5py, pyglet, hydra, flatdict
assert torch.version.cuda == "12.8", f"Expected CUDA 12.8 PyTorch build, got {torch.version.cuda}"
print("Simulator Python dependencies verified; no Kit/physics run performed.")
PY
fi

run "$SIM_PYTHON" -m venv "$ENV_ROOT/dataset"
if (( DRY_RUN )); then
    printf '%s\n' '+ add explicit simulator site-packages fallback to dataset .pth (dataset packages take precedence)'
else
    "$SIM_PYTHON" - "$DATASET_PYTHON" <<'PY'
import json
from pathlib import Path
import site
import subprocess
import sys
dataset_python = sys.argv[1]
target = Path(subprocess.check_output(
    [dataset_python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
paths = list(dict.fromkeys(str(Path(p).resolve()) for p in site.getsitepackages() if Path(p).is_dir()))
if not paths or any("\n" in p or "\r" in p for p in paths):
    raise SystemExit("Cannot safely create dataset simulator fallback paths")
(target / "_nepes_simulator_fallback.pth").write_text("\n".join(paths) + "\n")
print("Dataset fallback paths:", json.dumps(paths))
PY
fi
run "$DATASET_PYTHON" -m pip install -r "$RELEASE_ROOT/requirements/dataset.txt"
run "$DATASET_PYTHON" -m pip install --no-deps 'lerobot==0.4.4'
if (( DRY_RUN )); then
    printf '%s\n' '+ verify official LeRobot v3 API, PyAV H.264 encoder, and shared Torch import'
else
    "$DATASET_PYTHON" - <<'PY'
import av
import torch
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
assert CODEBASE_VERSION == "v3.0", CODEBASE_VERSION
assert av.Codec("libx264", "w").is_encoder
print("Official LeRobot v3 / PyAV exporter verified; torch", torch.__version__)
PY
fi
setup_ik_env "$IK_BASE_PYTHON"
if (( DRY_RUN )); then
    printf '+ write activation file: %s/env.sh\n' "$ENV_ROOT"
    exit 0
fi
{
    printf 'export NEPES_SIM_PYTHON=%q\n' "$SIM_PYTHON"
    printf 'export NEPES_DATASET_PYTHON=%q\n' "$DATASET_PYTHON"
    printf 'export NEPES_IK_PYTHON=%q\n' "$IK_PYTHON"
} > "$ENV_ROOT/env.sh"
printf '\nSetup complete. Activate with:\n  source %q\n' "$ENV_ROOT/env.sh"
