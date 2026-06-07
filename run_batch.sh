#!/usr/bin/env bash
# cat/dog all-forward pairs x 5 seeds (seed0 .. seed4)
# pair-mode: each image A with every later image B (cat_1→cat_2..dog_11, ...)
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
OUT_DIR="${SCRIPT_DIR}/out_batch2"
NUM_PROMPTS=5
NUM_SEEDS=5
# Comma-separated GPU ids; each GPU runs one job at a time
GPUS="${GPUS:-0,1,2,3}"
# ================

exec "$PYTHON" "${SCRIPT_DIR}/run_batch.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$OUT_DIR" \
  -n "$NUM_PROMPTS" \
  --num-seeds "$NUM_SEEDS" \
  --gpus "$GPUS" \
  "$@"
