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
  echo "ERROR: Python ${PYTHON_VERSION} required (Isaac Sim wheels are cp310-only)"
  exit 1
}

[ -d "OmniGibson" ] || {
  echo "ERROR: OmniGibson directory not found"
  exit 1
}

# Isaac Sim env: on aarch64 we REUSE the source-built Isaac Sim 5.1 instead of
# downloading x86_64 cp310 wheels, so ISAAC_PATH/EXP_PATH must be set (not an error).
export ISAAC_PATH="${ISAAC_PATH:-/home/batman/Documents/open-source/isaacsim/_build/linux-aarch64/release}"
export EXP_PATH="${EXP_PATH:-$ISAAC_PATH/apps}"
if [[ ! -d "$ISAAC_PATH" ]]; then
  echo "ERROR: ISAAC_PATH does not exist: $ISAAC_PATH"
  exit 1
fi
echo "Reusing source-built Isaac Sim at ISAAC_PATH=$ISAAC_PATH"

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
# Isaac Sim installation -- SKIPPED on aarch64
# =========================
# The original block here downloaded cp310 x86_64 Isaac Sim 4.5.0 wheels from
# pypi.nvidia.com. On this aarch64 (DGX Spark / GB10) box we instead REUSE the
# source-built Isaac Sim 5.1.0 at $ISAAC_PATH (exported above). No aarch64 Isaac
# 4.5 wheels exist, and Isaac 5.1 is cp311. Skipping the wheel download entirely.
echo "Skipping x86_64 Isaac Sim wheel download; reusing source build at $ISAAC_PATH"

# =========================
# Verify (wire source-built Isaac env; non-fatal so later installs still run)
# =========================
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