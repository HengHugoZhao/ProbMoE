# ProbMoE Finetuning for Qwen1.5-MoE

This folder contains the **ProbMoE** finetuning pipeline for the **Qwen1.5-MoE-A2.7B** base model. It is built on [LLaMA-Factory](LLaMA-Factory/) and provides fixed-`k` (*exact*) runs for GSM, Law, Summary, Translation, and CodeAlpaca, plus dynamic-`k` (*band-k*) runs for GSM, Law, and Translation.

---

## Directory Layout

```
train/qwen/
├── LLaMA-Factory/                                  # Modified LLaMA-Factory training stack
│   ├── examples/train_full/qwen1.5moe/full/        # Per-task YAML configs
│   │   ├── qwen1.5_gsm_lr3e-6.yaml
│   │   ├── qwen1.5_gsm_lr3e-6_dynamic.yaml
│   │   ├── qwen1.5_esft_law_lr1e-5.yaml
│   │   ├── qwen1.5_esft_summary_lr1e-6.yaml
│   │   ├── qwen1.5_esft_translation_lr5e-6.yaml
│   │   ├── qwen1.5_codealpaca_lr5e-6.yaml
│   │   └── ...
│   ├── src/llamafactory/model/Qwen/                 # Framework-local ProbMoE blocks
│   └── run/qwen1.5/probmoe_full/                    # Per-task launch scripts
│       ├── train_gsm.sh / train_gsm_dynamic.sh
│       ├── train_law.sh / train_law_dynamic.sh
│       ├── train_summary.sh
│       ├── train_translation.sh / train_translation_dynamic.sh
│       └── train_code.sh
└── scripts/
    ├── train_ProbMoE_qwen_exact.sh   # Fixed-k entry point
    └── train_ProbMoE_qwen_dynamic.sh # Dynamic-k entry point
```

---

## 1. Environment Setup

ProbMoE training for Qwen requires Python 3.12 and pinned versions of `transformers` and `deepspeed`.

```bash
# Create and activate the conda environment
conda create -n lmfact_env python=3.12 -y
conda activate lmfact_env

# Install LLaMA-Factory and dependencies
cd train/qwen/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install transformers==4.51.3
pip install deepspeed==0.16.7
pip install wandb
```


---

## 2. Per-Task Configuration

Training is driven by two layers:

1. **YAML configs** under [LLaMA-Factory/examples/train_full/qwen1.5moe/full/](LLaMA-Factory/examples/train_full/qwen1.5moe/full/) — define model, dataset, optimizer, and ProbMoE flags.
2. **Shell launchers** under [LLaMA-Factory/run/qwen1.5/probmoe_full/](LLaMA-Factory/run/qwen1.5/probmoe_full/) — pick the YAML, set `CUDA_VISIBLE_DEVICES`, and call `llamafactory-cli train`.

| YAML | Routing Mode | Purpose |
|---|---|---|
| `qwen1.5_<task>_lr<lr>.yaml` | Fixed `k` | Train with a constant number of active experts |
| `qwen1.5_<task>_lr<lr>_dynamic.yaml` | Band `k` | Train with `k` sampled in `[min_k, max_k]` per step |

### Required edits before launching

**(a) In the YAML config** (e.g. [qwen1.5_gsm_lr3e-6.yaml](LLaMA-Factory/examples/train_full/qwen1.5moe/full/qwen1.5_gsm_lr3e-6.yaml)):

| Field | Description |
|---|---|
| `model_name_or_path` | Base model — defaults to `Qwen/Qwen1.5-MoE-A2.7B` |
| `probmoe` | Enable ProbMoE routing (`true`) |
| `band_k` | `true` for dynamic-k, `false` for fixed-k |
| `min_k` / `max_k` | Dynamic-k bounds (used when `band_k: true`) |
| `dataset` | LLaMA-Factory dataset key (e.g. `gsm_lf`) |
| `output_dir` / `run_name` | Checkpoint dir and W&B run name |
| `learning_rate` / `num_train_epochs` / `per_device_train_batch_size` | Standard training hyperparameters |
| `deepspeed` | DeepSpeed config — defaults to `ds_z3_config.json` |

The ProbMoE blocks live in [LLaMA-Factory/src/llamafactory/model/Qwen/](LLaMA-Factory/src/llamafactory/model/Qwen/) and are imported through the `llamafactory.model` package. No source-path field is required, and import or patch failures stop the run.

**(b) In the launcher shell script** (e.g. [train_gsm.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_gsm.sh)):

- Set `CUDA_VISIBLE_DEVICES` to your target GPUs (default `0,1,2,3`).
- The `CONFIG_FILES` array selects which YAML(s) to run — update if you renamed the config or want to sweep multiple LRs.

---

## 3. Launching a Training Run

The top-level launchers under [scripts/](scripts/) wrap the per-task scripts and handle cache/W&B environment variables.

### Fixed-k

```bash
cd train/qwen/scripts
export WANDB_API_KEY=...  # optional when W&B is disabled
bash train_ProbMoE_qwen_exact.sh gsm
```

### Dynamic-k

```bash
cd train/qwen/scripts
bash train_ProbMoE_qwen_dynamic.sh gsm
```

Pass the dataset as the first argument, or set `DATASET` in the environment. Dynamic-k launchers are currently provided for GSM, Law, and Translation.

Logs are written to `LLaMA-Factory/logs/train_full_<config>.log`.

### Verifying the ProbMoE patch

If ProbMoE was patched into the model successfully, you should see a log line of the form:

```
[INFO] Found ProbMoE block: model.layers.0.mlp (<class 'llamafactory.model.Qwen.ProbMoE_V1_qwen_exact.ProbMoEQwen2MoeSparseMoeBlock'>)
```
If you don't see this, check that `probmoe: true` is set in the YAML. A missing or mismatched ProbMoE block now raises an error during model loading.

---

## 4. Supported Datasets

| Dataset | Launcher (fixed-k) | Launcher (dynamic-k) | Task |
|---|---|---|---|
| `gsm` | [train_gsm.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_gsm.sh) | [train_gsm_dynamic.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_gsm_dynamic.sh) | Grade-school math reasoning |
| `law` | [train_law.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_law.sh) | [train_law_dynamic.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_law_dynamic.sh) | Legal-domain instruction following |
| `summary` | [train_summary.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_summary.sh) | — | Summarization |
| `translation` | [train_translation.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_translation.sh) | [train_translation_dynamic.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_translation_dynamic.sh) | Machine translation |
| `code` | [train_code.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_code.sh) | — | Code instruction following (CodeAlpaca) |

---

## 5. Evaluation

Evaluation lives in a separate folder. See [eval/README.md](../../eval/README.md) for task-specific evaluation instructions (GSM, MBPP, ESFT benchmarks).

---

## Troubleshooting

- **`transformers` / `deepspeed` import errors:** make sure you are on the pinned versions (`transformers==4.51.3`, `deepspeed==0.16.7`).
- **OOM:** lower `per_device_train_batch_size`, switch to `ds_z3_config.json` (already the default), or reduce `cutoff_len`.
- **W&B not logging:** confirm `WANDB_API_KEY` is exported before running `scripts/train_ProbMoE_qwen_exact.sh` or `scripts/train_ProbMoE_qwen_dynamic.sh`.
- **Cache permission errors:** set the Hugging Face and Triton cache environment variables to writable directories before launching.

---

## Citations

This work uses the following models, codebases, and datasets:

- **Qwen1.5-MoE-A2.7B** — base model used for finetuning.
  [Qwen/Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)
- **LLaMA-Factory** — finetuning framework adapted for ProbMoE training.
  [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- **ESFT** — project and datasets (Law, Summary, Translation) used for domain-specific finetuning and evaluation.
  [deepseek-ai/ESFT](https://github.com/deepseek-ai/ESFT)
- **GSM8K** — grade-school math reasoning benchmark (Cobbe et al., 2021).
  [openai/grade-school-math](https://github.com/openai/grade-school-math) · [Paper](https://arxiv.org/abs/2110.14168)
