#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-${DATASET:-gsm}}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPEN_INSTRUCT_DIR="${SCRIPT_DIR}/../open_instruct"
LAUNCHER="run/${DATASET}/train_${DATASET}_dynamic.sh"
LOG_DIR="${SCRIPT_DIR}/log/dynamic/${DATASET}"

# The translation launcher keeps its historical band-k filename.
if [[ "${DATASET}" == "translation" ]]; then
    LAUNCHER="run/translation/train_translation_band_k.sh"
fi

if [[ ! -f "${OPEN_INSTRUCT_DIR}/${LAUNCHER}" ]]; then
    echo "Unsupported dataset or missing dynamic-k launcher: ${OPEN_INSTRUCT_DIR}/${LAUNCHER}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${OPEN_INSTRUCT_DIR}"
bash "${LAUNCHER}" 2>&1 | tee "${LOG_DIR}/train_${DATASET}_ProbMoE_dynamic.log"
