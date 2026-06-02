CUDA_VISIBLE_DEVICES=0,1,2,3 python run_eval_gsm.py \
    --dataset_name RoxanneWsyw/gsm\
    --test_file test.jsonl \
    --save_dir simoe_result_gsm \
    --model_name_or_path allenai/OLMoE-1B-7B-0125 \
    --tokenizer_name_or_path allenai/OLMoE-1B-7B-0125 \
    --eval_batch_size 256 \
    --model_type "olmoe" \
    --mode "exact" #or band for the dynamic MoE block variant