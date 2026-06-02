DATASET="gsm" #enter the dataset name here, e.g. gsm, law, etc.

export WANDB_API_KEY=""

cd /ProbMoE/train/olmoe/open_instruct #enter the path to the open_instruct directory here

mkdir -p log/exact/${DATASET}

bash run/${DATASET}/train_${DATASET}.sh 2>&1 | tee /ProbMoE/train/olmoe/script/log/exact/${DATASET}/train_${DATASET}_ProbMoE_exact.log