# Evaluation

Task-specific evaluation scripts for ProbMoE checkpoints. When a custom routing class is required, the evaluation helper imports the same framework-local OLMoE and Qwen blocks used for training; no external source path is required.

---

## GSM8K

```bash
pip install evaluate
MODEL_PATH=/path/to/checkpoint \
MODEL_TYPE=olmoe \
MODE=exact \
bash eval/gsm/run_eval_gsm.sh
```

Use `MODEL_TYPE=qwen` for Qwen checkpoints and `MODE=band` for dynamic-k checkpoints.

## ESFT (Law, Summary, Translation)

Use the `exact/` scripts for fixed-*k* checkpoints and `dynamic/` for Dynamic-*k* checkpoints.

```bash
MODEL_PATH=/path/to/checkpoint \
EVAL_DATASETS=summary \
OPENAI_API_KEY=... \
bash eval/esft/exact/eval_esft.sh

MODEL_PATH=/path/to/checkpoint \
EVAL_DATASETS=law \
OPENAI_API_KEY=... \
bash eval/esft/dynamic/eval_esft_band.sh
```

## MBPP / HumanEval (CodeAlpaca)

Run with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

cd /path/to/ProbMoE
BASE_DIR=/path/to/checkpoints bash eval/mbpp/lm_eval_code.sh
```

---

## Acknowledgments

Evaluation inherited from [DenseMixer/experiments/open-instruct/eval](https://github.com/yaof20/DenseMixer/tree/main/experiments/open-instruct/eval).
