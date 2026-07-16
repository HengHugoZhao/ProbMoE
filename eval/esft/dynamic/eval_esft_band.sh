#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../../.."
: "${MODEL_PATH:?Set MODEL_PATH to a dynamic-k OLMoE checkpoint.}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -m eval.esft.dynamic.eval_esft_band \
    --eval_datasets="${EVAL_DATASETS:-law}" \
    --model_path="${MODEL_PATH}" \
    --output_dir="${OUTPUT_DIR:-results/esft/dynamic}" \
    --max_new_tokens=512 \
    --openai_api_key="${OPENAI_API_KEY:-}" \
    --eval_batch_size=2 \
    --gpu_count 1
