#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
MODEL_PATH="${MODEL_PATH:-allenai/OLMoE-1B-7B-0125}"
MODEL_TYPE="${MODEL_TYPE:-olmoe}"
MODE="${MODE:-exact}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" python -m eval.gsm.run_eval_gsm \
    --dataset_name RoxanneWsyw/gsm \
    --test_file test.jsonl \
    --save_dir results/gsm \
    --model_name_or_path "${MODEL_PATH}" \
    --tokenizer_name_or_path "${MODEL_PATH}" \
    --eval_batch_size 256 \
    --model_type "${MODEL_TYPE}" \
    --mode "${MODE}"
