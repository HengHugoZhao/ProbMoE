#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-${DATASET:-gsm}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DIR="${SCRIPT_DIR}/../LLaMA-Factory"
LAUNCHER="run/qwen1.5/probmoe_full/train_${DATASET}_dynamic.sh"

if [[ ! -f "${LLAMA_FACTORY_DIR}/${LAUNCHER}" ]]; then
    echo "Unsupported dataset or missing dynamic-k launcher: ${LLAMA_FACTORY_DIR}/${LAUNCHER}" >&2
    exit 1
fi

cd "${LLAMA_FACTORY_DIR}"
bash "${LAUNCHER}"
