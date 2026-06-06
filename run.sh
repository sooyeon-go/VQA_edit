#!/usr/bin/env bash
# VQA -> LLM -> Hunyuan i2i (full pipeline)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-qwen}"

# conda env python 우선 사용 (없으면 PYTHON 또는 시스템 python)
if [[ -z "${PYTHON:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    if [[ -x "${CONDA_BASE}/envs/${ENV_NAME}/bin/python" ]]; then
      PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
    else
      PYTHON="python"
    fi
  else
    PYTHON="python"
  fi
fi

# ===== 여기만 수정 =====
IMAGE_A="/path/to/start.png"
IMAGE_B="/path/to/end.png"
NUM_PROMPTS=5
OUT_DIR="${SCRIPT_DIR}/out"

MODEL_ROOT="/data/shared-vilab/pretrained_models"
VQA_MODEL="${MODEL_ROOT}/Qwen3-VL-8B-Instruct"
LLM_MODEL="${MODEL_ROOT}/qwen3-32b-weights"
HUNYUAN_MODEL="${MODEL_ROOT}/HunyuanImage-3-Instruct"
# =======================

usage() {
  cat <<EOF
Usage: $(basename "$0") [extra run_pipeline.py options]

Edit IMAGE_A, IMAGE_B, NUM_PROMPTS in this script before running.

Environment:
  ENV_NAME=${ENV_NAME}   conda env for python (default: qwen)
  PYTHON                 override python path

Example:
  $(basename "$0") --chain-mode sequential --seed 42
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$IMAGE_A" ]]; then
  echo "Error: IMAGE_A not found: $IMAGE_A" >&2
  echo "Edit run.sh and set IMAGE_A to your start image." >&2
  exit 1
fi
if [[ ! -f "$IMAGE_B" ]]; then
  echo "Error: IMAGE_B not found: $IMAGE_B" >&2
  echo "Edit run.sh and set IMAGE_B to your end image." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "python:  $PYTHON" >&2
echo "image A: $IMAGE_A" >&2
echo "image B: $IMAGE_B" >&2
echo "steps:   $NUM_PROMPTS" >&2
echo "out:     $OUT_DIR" >&2
echo "vqa:     $VQA_MODEL" >&2
echo "llm:     $LLM_MODEL" >&2
echo "hunyuan: $HUNYUAN_MODEL" >&2

exec "$PYTHON" "${SCRIPT_DIR}/run_pipeline.py" \
  --image-a "$IMAGE_A" \
  --image-b "$IMAGE_B" \
  -n "$NUM_PROMPTS" \
  --out-dir "$OUT_DIR" \
  --vqa-model "$VQA_MODEL" \
  --llm-model "$LLM_MODEL" \
  --hunyuan-model "$HUNYUAN_MODEL" \
  "$@"
