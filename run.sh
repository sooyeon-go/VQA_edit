#!/usr/bin/env bash
# VQA -> LLM -> Hunyuan i2i (full pipeline)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# ===== 여기만 수정 =====
IMAGE_A="/path/to/start.png"
IMAGE_B="/path/to/end.png"
NUM_PROMPTS=5
OUT_DIR="/mnt/sy/qwen/out"
# =======================

exec "$PYTHON" "${SCRIPT_DIR}/run_pipeline.py" \
  --image-a "$IMAGE_A" \
  --image-b "$IMAGE_B" \
  -n "$NUM_PROMPTS" \
  --out-dir "$OUT_DIR" \
  "$@"
