#!/usr/bin/env bash
# cat/dog circular pairs x 5 seeds (seed0 .. seed4)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-qwen}"

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

# ===== 설정 =====
DATASET_DIR="${SCRIPT_DIR}/image_dataset"
OUT_DIR="${SCRIPT_DIR}/out_batch"
NUM_PROMPTS=5
NUM_SEEDS=5
GPU_ID="${GPU_ID:-0}"
# ================

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec "$PYTHON" "${SCRIPT_DIR}/run_batch.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$OUT_DIR" \
  -n "$NUM_PROMPTS" \
  --num-seeds "$NUM_SEEDS" \
  --gpu 0 \
  "$@"
