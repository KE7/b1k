#!/usr/bin/env bash
set -e

# =========================
# Config
# =========================
CUDA_VERSION="12.4"
PYTHON_VERSION="3.11" # Isaac Sim 5.1 (kit 107, aarch64 source build) is cp311
WORKDIR=$(pwd)

# Optional flags
DATASET=false
ACCEPT_DATASET_TOS=false

export OMNI_KIT_ACCEPT_EULA=YES


# =========================
# Parse arguments
# =========================
HELP=false
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help) HELP=true; shift ;;
    --dataset) DATASET=true; shift ;;
    --accept-dataset-tos) ACCEPT_DATASET_TOS=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [ "$HELP" = true ]; then
  cat << EOF
Usage: ./uv_install.sh [OPTIONS]

Options:
  -h, --help              Show this help
  --dataset               Download OmniGibson robot assets + BEHAVIOR-1K assets + 2025 challenge instances
  --accept-dataset-tos    Auto-accept BEHAVIOR dataset license (passed to download_behavior_1k_assets)

Example:
  ./uv_install.sh --dataset --accept-dataset-tos
EOF
  exit 0
fi


# =========================
# Sanity checks
# =========================
command -v uv >/dev/null || {
  echo "ERROR: uv not found. Install with: pip install uv"
  exit 1
}

python --version | grep -q "Python ${PYTHON_VERSION}" || {
  echo "ERROR: Python ${PYTHON_VERSION} required (Isaac Sim 5.1 source build is cp311)"
  exit 1
}

[ -d "OmniGibson" ] || {
  echo "ERROR: OmniGibson directory not found"
  exit 1
}

# =========================
# Architecture gate / dispatch
# =========================
# Unified installer for OmniGibson 3.8.0 / Isaac Sim 5.1 / Python 3.11 (cp311) on
# BOTH aarch64 and x86_64. The two arches differ ONLY in how Isaac Sim 5.1 is
# obtained:
#   - aarch64 (DGX Spark / GB10): REUSE a source-built Isaac Sim 5.1 release tree at
#     $ISAAC_PATH. No aarch64 Isaac Sim wheels are published, so it is built from
#     source and reused here.
#   - x86_64: install the published Isaac Sim 5.1 cp311 wheels from pypi.nvidia.com
#     ("isaacsim[all,extscache]==5.1.0"). manylinux_2_35_x86_64 / cp311.
# Both arches converge on Isaac Sim 5.1 / cp311, so PYTHON_VERSION="3.11" (Config,
# above) is correct for both.
ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64) ISAAC_ARCH="aarch64" ;;
  x86_64)        ISAAC_ARCH="x86_64" ;;
  *)
    echo "ERROR: Unsupported architecture: $ARCH (expected aarch64/arm64 or x86_64)."
    exit 1
    ;;
esac

if [[ "$ISAAC_ARCH" == "x86_64" ]]; then
  # x86_64 installs Isaac Sim 5.1 wheels, which manage ISAAC_PATH/EXP_PATH/CARB_APP_PATH
  # themselves on import. A pre-set Isaac Sim env would collide with the wheel install,
  # so refuse it (mirrors the upstream x86_64 wheel installer's env-conflict check).
  if [[ -n "${EXP_PATH:-}" || -n "${CARB_APP_PATH:-}" || -n "${ISAAC_PATH:-}" ]]; then
    echo "ERROR: Existing Isaac Sim environment variables detected (EXP_PATH/CARB_APP_PATH/ISAAC_PATH)."
    echo "       The x86_64 wheel install manages these itself; unset them and re-run."
    exit 1
  fi
  echo "x86_64 detected: will install Isaac Sim 5.1 cp311 wheels from pypi.nvidia.com."
fi

if [[ "$ISAAC_ARCH" == "aarch64" ]]; then
# Isaac Sim env: on aarch64 we REUSE the source-built Isaac Sim 5.1 instead of
# downloading x86_64 cp310 wheels, so ISAAC_PATH must point at a source-built Isaac
# Sim 5.1 release tree. We do NOT hardcode any machine-specific/home path here.
# Resolution order:
#   1. ISAAC_PATH from the environment (preferred -- fully overridable + portable).
#   2. Auto-detect a source build at a standard repo-relative location, if present.
#   3. Otherwise FAIL with a clear, actionable message.
if [[ -z "${ISAAC_PATH:-}" ]]; then
  for _cand in \
    "$WORKDIR/isaacsim/_build/linux-aarch64/release" \
    "$WORKDIR/../isaacsim/_build/linux-aarch64/release"; do
    if [[ -d "$_cand" ]]; then
      ISAAC_PATH="$(cd "$_cand" && pwd)"
      echo "Auto-detected source-built Isaac Sim at ISAAC_PATH=$ISAAC_PATH"
      break
    fi
  done
fi
if [[ -z "${ISAAC_PATH:-}" ]]; then
  echo "ERROR: ISAAC_PATH is not set and no source-built Isaac Sim was auto-detected."
  echo "       On aarch64 we reuse a source-built Isaac Sim 5.1 release tree instead of"
  echo "       downloading x86_64 wheels. Set ISAAC_PATH to your Isaac Sim release dir, e.g.:"
  echo "         export ISAAC_PATH=/path/to/isaacsim/_build/linux-aarch64/release"
  echo "       then re-run: ./uv_install.sh"
  exit 1
fi
export ISAAC_PATH
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
if [[ ! -d "$ISAAC_PATH" ]]; then
  echo "ERROR: ISAAC_PATH does not exist: $ISAAC_PATH"
  exit 1
fi
echo "Reusing source-built Isaac Sim at ISAAC_PATH=$ISAAC_PATH"
fi

# =========================
# Initialize uv project
# =========================
if [ ! -f pyproject.toml ]; then
  echo "Initializing uv project..."
  uv init
fi


# =========================
# Install OmniGibson (editable)
# =========================
echo "Installing OmniGibson (editable)..."
uv pip install -e "$WORKDIR/bddl3"
uv pip install -e "$WORKDIR/OmniGibson"

# =========================
# Isaac Sim 5.1 installation (arch-dependent)
# =========================
# Both arches target Isaac Sim 5.1 / cp311 (OmniGibson 3.8.0); they differ only in
# how Isaac Sim is obtained:
#   - aarch64: REUSE the source-built Isaac Sim 5.1.0 at $ISAAC_PATH (resolved above).
#     No aarch64 Isaac Sim wheels are published, so the wheel download is skipped.
#   - x86_64: install the published Isaac Sim 5.1 cp311 wheels from pypi.nvidia.com.
if [[ "$ISAAC_ARCH" == "aarch64" ]]; then
  echo "Skipping x86_64 Isaac Sim wheel download; reusing source build at $ISAAC_PATH"
else
  echo "Installing Isaac Sim 5.1 (cp311 wheels) from pypi.nvidia.com..."
  # NVIDIA's documented Isaac Sim 5.1 pip install: the "isaacsim" meta-package with
  # the [all,extscache] extras pulls every isaacsim.* component + the extscache
  # bundles (the same component set the aarch64 source build provides). cp311 /
  # manylinux_2_35_x86_64 wheels are published on pypi.nvidia.com. NOTE: the 5.1
  # wheels are manylinux_2_35, so they require glibc >= 2.35 (Ubuntu 22.04+); unlike
  # the old 4.5 (manylinux_2_34) block there is no manylinux_2_31 fallback rename.
  uv pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

  # =========================
  # Fix websockets conflict
  # =========================
  # The Isaac extscache bundles a pip_prebundle websockets that collides with the one
  # OmniGibson/uv installs. Remove the bundled copy (same cleanup the original x86_64
  # wheel install performed). ISAAC_PATH is set as a side effect of importing the
  # freshly-installed isaacsim package.
  ISAAC_PATH=$(python - << 'EOF'
import isaacsim, os
print(os.environ.get("ISAAC_PATH", ""))
EOF
)
  if [ -n "$ISAAC_PATH" ] && [ -d "$ISAAC_PATH/extscache" ]; then
    echo "Fixing websockets conflict..."
    find "$ISAAC_PATH/extscache" \
      -type d \
      -path "*/pip_prebundle/websockets" \
      -exec rm -rf {} + || true
  fi
fi

# =========================
# Verify
# =========================
if [[ "$ISAAC_ARCH" == "aarch64" ]]; then
  # aarch64: wire the source-built Isaac env (non-fatal so later installs still run)
  (
    export CARB_APP_PATH="$ISAAC_PATH/kit"
    source "$ISAAC_PATH/setup_python_env.sh"
    export LD_PRELOAD="$ISAAC_PATH/kit/libcarb.so"
    python - << 'EOF'
import omnigibson
print("omnigibson", omnigibson.__version__)
import isaacsim
print("✓ OmniGibson and Isaac Sim importable")
EOF
  ) || echo "WARN: import verify failed (continuing; will re-check in Phase 3 with full env wiring)"
else
  # x86_64: the wheels install a self-contained Isaac Sim env, so a plain import
  # verifies the install (non-fatal so later installs still run).
  python - << 'EOF' || echo "WARN: import verify failed (continuing)"
import omnigibson
print("omnigibson", omnigibson.__version__)
import isaacsim
print("✓ OmniGibson and Isaac Sim importable")
EOF
fi

echo ""
echo "=== OmniGibson + Isaac Sim (uv) installation complete ==="

# =========================
# Datasets (optional)
# =========================
if [ "$DATASET" = true ]; then
  # Ensure we accept Isaac EULA for any OmniKit-backed downloads
  export OMNI_KIT_ACCEPT_EULA=YES

  # Ensure OmniGibson is importable in the current environment
  python -c "import omnigibson" >/dev/null 2>&1 || {
    echo "ERROR: OmniGibson import failed. Make sure OmniGibson is installed in the active venv."
    exit 1
  }

  echo "Installing datasets..."

  if [ "$ACCEPT_DATASET_TOS" = true ]; then
    DATASET_ACCEPT_FLAG="True"
  else
    DATASET_ACCEPT_FLAG="False"
  fi

  echo "Downloading OmniGibson robot assets..."
  set -euo pipefail
  # 0) Resolve OmniGibson DATA_PATH from the uv environment
  DATA_PATH="$(python - <<'PY'
from omnigibson.macros import gm
print(gm.DATA_PATH)
PY
)"
  ASSETS_DIR="${DATA_PATH}/omnigibson-robot-assets"
  CUSTOM_REL="models/r1pro/urdf/r1pro_ik.urdf"
  CUSTOM_SRC="${ASSETS_DIR}/${CUSTOM_REL}"

  STAMP="$(date +%Y%m%d_%H%M%S)"
  STASH_DIR="${DATA_PATH}/_custom_overlays_${STAMP}"
  # 1) Stash the custom file (if it exists)
  mkdir -p "${STASH_DIR}/$(dirname "${CUSTOM_REL}")"

  if [ -f "${CUSTOM_SRC}" ]; then
    echo "Stashing custom file..."
    cp -a "${CUSTOM_SRC}" "${STASH_DIR}/${CUSTOM_REL}"
  else
    echo "WARNING: r1pro_ik.urdf file not found at ${CUSTOM_SRC}"
    echo "         Continuing anyway (will just reinstall assets). but you need to download it manually"
  fi

  rm -rf "${ASSETS_DIR}"

  python -c "from omnigibson.utils.asset_utils import download_omnigibson_robot_assets; download_omnigibson_robot_assets()" || {
    echo "ERROR: OmniGibson robot assets installation failed"
    exit 1
  }

  # 4) Restore (overlay) the custom file back into the new install
  if [ -f "${STASH_DIR}/${CUSTOM_REL}" ]; then
    echo "Restoring custom file into fresh install..."
    mkdir -p "${ASSETS_DIR}/$(dirname "${CUSTOM_REL}")"
    cp -a "${STASH_DIR}/${CUSTOM_REL}" "${CUSTOM_SRC}"
    echo "✓ Restored: ${CUSTOM_SRC}"
  fi

  # 5) Copy r1pro_ik.urdf from the repo if it doesn't exist after restore
  #    This URDF has mobile-base and gripper joints fixed for IK-only use.
  if [ ! -f "${CUSTOM_SRC}" ]; then
    REPO_IK_URDF="${WORKDIR}/assets/r1pro_ik.urdf"
    if [ -f "${REPO_IK_URDF}" ]; then
      echo "Copying r1pro_ik.urdf from repo..."
      mkdir -p "${ASSETS_DIR}/$(dirname "${CUSTOM_REL}")"
      cp -a "${REPO_IK_URDF}" "${CUSTOM_SRC}"
      echo "✓ Installed: ${CUSTOM_SRC}"
    else
      echo "WARNING: r1pro_ik.urdf not found at ${REPO_IK_URDF}"
      echo "         BEHAVIOR R1Pro environments will not work without this file."
    fi
  fi

  echo "Downloading BEHAVIOR-1K assets..."
  python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=${DATASET_ACCEPT_FLAG})" || {
    echo "ERROR: BEHAVIOR-1K assets installation failed"
    exit 1
  }

  echo "Downloading 2025 BEHAVIOR Challenge Task Instances..."
  python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()" || {
    echo "ERROR: 2025 BEHAVIOR Challenge Task Instances installation failed"
    exit 1
  }

  echo "✓ Dataset installation completed"
fi

# =========================
# install curobo
# =========================
export GIT_LFS_SKIP_SMUDGE=1
uv pip install nvidia_curobo@git+https://github.com/StanfordVL/curobo@cbaf7d32436160956dad190a9465360fad6aba73

# =========================
# reinstall pyroki
# =========================
uv pip install pyroki@git+https://github.com/chungmin99/pyroki.git

# =========================
# Fix PyTorch CUDA compatibility
# =========================
# OmniGibson may install a newer PyTorch (cu130) that requires driver 570+.
# Downgrade to cu124 if the installed torch targets CUDA 13.0 but the driver
# only supports CUDA 12.x (driver < 570).
TORCH_CUDA=$(python - 2>/dev/null <<'PY'
try:
    import torch
    print(torch.version.cuda or "")
except Exception:
    print("")
PY
)
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1 || echo "0")
if [[ "${TORCH_CUDA}" == "13.0" && "${DRIVER_MAJOR:-0}" -lt 570 ]]; then
  echo "Downgrading PyTorch to cu124 (driver ${DRIVER_MAJOR} < 570, torch.version.cuda=${TORCH_CUDA})..."
  uv pip install "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" \
    --extra-index-url https://download.pytorch.org/whl/cu124
fi

# =========================
# Perception server dependencies
# =========================
# SAM3 and ContactGraspNet are perception servers used by BEHAVIOR task configs.
# Install SAM3 from the vendored submodule and its runtime dependencies.
CAPX_ROOT="$(cd "${WORKDIR}/../.." && pwd)"

if [ -d "${CAPX_ROOT}/capx/third_party/sam3" ]; then
  echo "Installing SAM3 perception server..."
  uv pip install "${CAPX_ROOT}/capx/third_party/sam3" --no-deps
  uv pip install iopath einops timm "ftfy==6.1.1" decord pycocotools
fi

# pyrender is required by ContactGraspNet scene renderer
uv pip install pyrender

# open3d is required by capx integrations (FrankaControlApi)
uv pip install open3d