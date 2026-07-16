#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../../.."
: "${MODEL_PATH:?Set MODEL_PATH to an exact-k ProbMoE checkpoint.}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" python -m eval.esft.exact.eval_esft \
    --eval_datasets="${EVAL_DATASETS:-summary}" \
    --model_path="${MODEL_PATH}" \
    --output_dir="${OUTPUT_DIR:-results/esft/exact}" \
    --max_new_tokens=512 \
    --openai_api_key="${OPENAI_API_KEY:-}" \
    --eval_batch_size=2 \
    --gpu_count 2
