#!/usr/bin/env bash
# Create conda env and install dependencies for run_pipeline.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-qwen}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CUDA_INDEX="${CUDA_INDEX:-cu128}"   # Hunyuan official: cu128
INSTALL_FLASHINFER="${INSTALL_FLASHINFER:-0}"

echo "==> env name:     $ENV_NAME"
echo "==> python:       $PYTHON_VERSION"
echo "==> cuda wheels:  $CUDA_INDEX"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> using existing conda env: $ENV_NAME"
else
  echo "==> creating conda env: $ENV_NAME"
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
fi

conda activate "$ENV_NAME"

echo "==> upgrading pip"
python -m pip install -U pip wheel setuptools

echo "==> installing PyTorch (${CUDA_INDEX})"
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}"

echo "==> installing Python dependencies"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

if [[ "$INSTALL_FLASHINFER" == "1" ]]; then
  echo "==> installing flashinfer (optional, faster Hunyuan MoE)"
  python -m pip install flashinfer-python==0.5.0 || {
    echo "Warning: flashinfer install failed; use --moe-impl eager" >&2
  }
fi

echo
echo "Done."
echo
echo "Activate:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Download models:"
echo "  python ${SCRIPT_DIR}/model_download.py"
echo
echo "Run pipeline (edit paths in run.sh first):"
echo "  ${SCRIPT_DIR}/run.sh"
echo
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY
