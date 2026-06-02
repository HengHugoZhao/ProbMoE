#!/bin/bash

# mkdir -p logs
log_file="logs/eval_single_law_$(date '+%Y%m%d_%H%M%S').log"


export CUDA_VISIBLE_DEVICES=0
python eval_esft_band.py \
    --eval_datasets=law \
    --model_path= \
    --output_dir=results  \
    --max_new_tokens=512 \
    --openai_api_key="" \
    --eval_batch_size=2 \
    --gpu_count 1 \