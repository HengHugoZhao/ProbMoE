#!/bin/bash

# mkdir -p logs
log_file="logs/eval_single_$(date '+%Y%m%d_%H%M%S').log"


export CUDA_VISIBLE_DEVICES=0,1
python eval_esft.py \
    --eval_datasets=summary \
    --model_path= \
    --output_dir=results \
    --max_new_tokens=512 \
    --openai_api_key="" \
    --eval_batch_size=2 \
    --gpu_count 2 \