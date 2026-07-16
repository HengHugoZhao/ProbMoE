#!/bin/bash
MAX_K=8
MIN_K=6

bash finetune/script/finetune/full_simple.sh \
    --task law \
    --model olmoe \
    --total_batch_size 256 \
    --num_train_epochs 8 \
    --num_gpus 4 \
    --devices 4,5,6,7 \
    --port 28000 \
    --lr 2e-5 \
    --lr_scheduler_type "linear" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_checkpointing False \
    --probmoe True \
    --v2 False \
    --band_k True \
    --do_eval True \
    --max_k "$MAX_K" \
    --min_k "$MIN_K" \
    --seed 42 \
    --freeze_gate False \
    --output_suffix "LAW_ProbMoE_FT_Dynamic_K_${MAX_K}_${MIN_K}_seed42"
