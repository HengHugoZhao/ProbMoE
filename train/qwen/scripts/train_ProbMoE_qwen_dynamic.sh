DATASET="gsm" #change this if you want to use other dataset

export WANDB_API_KEY=""

cd #to the directory of LLaMA-Factory

bash run/qwen1.5/probmoe_full/train_${DATASET}_dynamic.sh