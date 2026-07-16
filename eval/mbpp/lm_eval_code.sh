#!/usr/bin/env bash
set -euo pipefail

: "${BASE_DIR:?Set BASE_DIR to the directory containing the checkpoints.}"

model_list=()
for epoch in {0..3}; do
    model_list+=("${BASE_DIR}/epoch_${epoch}") # change to checkpoint-* for qwen
done
# model_list+=("${BASE_DIR}/epoch_0")
export HF_ALLOW_CODE_EVAL="1"

task_log="evaluation_code_8-e5_temp1e-1_8epoch_com$(date '+%Y%m%d_%H%M%S').log"
required_gpus=2


> "$task_log"

tasks=("mbpp" "humaneval")

for model_nm in "${model_list[@]}"; do
    echo "Evaluating model: $model_nm"

    for task in "${tasks[@]}"; do
        echo "Running task: $task"

        CUDA_VISIBLE_DEVICES=0,1  accelerate launch \
            --num_processes $required_gpus \
            --multi_gpu \
            --mixed_precision bf16 \
            -m lm_eval run \
            --model hf \
            --model_args "pretrained=$model_nm" \
            --output_path "code_result/res_${task}_$(basename "$model_nm")" \
            --log_samples \
            --task "$task" \
            --batch_size 16 \
            --device "cuda" \
            --confirm_run_unsafe_code 2>&1 | tee -a "$task_log"
    done
done

