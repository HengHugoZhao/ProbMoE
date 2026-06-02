# Evaluation

Task-specific evaluation scripts for SIMoE checkpoints. Set `--mode exact` for the fixed-*k* variant and `--mode band` for the Dynamic-*k* variant where applicable.

---

## GSM8K

```bash
pip install evaluate
cd gsm
# Edit run_eval_gsm.sh: set --model_name_or_path, --model_type ("olmoe" or "qwen"),
# and --mode ("exact" or "band").
bash run_eval_gsm.sh
```

## ESFT (Law, Summary, Translation)

Use the `exact/` scripts for fixed-*k* checkpoints and `dynamic/` for Dynamic-*k* checkpoints.

```bash
cd esft/exact          # or: cd esft/dynamic
# Edit eval_esft.sh (or eval_esft_band.sh): set --model_path, --eval_datasets,
# and --openai_api_key (used for LLM-judged scoring).
bash eval_esft.sh      # or: bash eval_esft_band.sh
```

## MBPP / HumanEval (CodeAlpaca)

Run with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

cd ../mbpp
# Edit lm_eval_code.sh: set BASE_DIR to the checkpoint directory.
bash lm_eval_code.sh
```

---

## Acknowledgments

Evaluation inherited from [DenseMixer/experiments/open-instruct/eval](https://github.com/yaof20/DenseMixer/tree/main/experiments/open-instruct/eval).
