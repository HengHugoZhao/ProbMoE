# ProbMoE Finetuning for Qwen1.5-MoE

This folder contains the finetuning pipeline for **ProbMoE** (Sparse-Interpolated Mixture of Experts) on top of the **Qwen1.5-MoE-A2.7B** base model. It is built on top of [LLaMA-Factory](LLaMA-Factory/) and supports multiple downstream tasks (GSM, Law, Summary, Translation, CodeAlpaca) with both fixed-`k` (*exact*) and dynamic-`k` (*band-k*) expert routing.

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
│   └── run/qwen1.5/probmoe_full/                    # Per-task launch scripts
│       ├── train_gsm.sh / train_gsm_dynamic.sh
│       ├── train_law.sh / train_law_dynamic.sh
│       ├── train_summary.sh
│       ├── train_translation.sh / train_translation_dynamic.sh
│       └── train_code.sh
└── scripts/
    ├── train_ProbMoE_exact_qwen.sh                   # Fixed-k entry point
    └── train_ProbMoE_exact_qwen _dynamic.sh          # Dynamic-k entry point
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
| `custom_moe_path` | **Absolute** path to `ProbMoE/models` (e.g. `/u/qgg5se/ProbMoE/models`) — ⚠️ **empty by default, must be filled** |
| `dataset` | LLaMA-Factory dataset key (e.g. `gsm_lf`) |
| `output_dir` / `run_name` | Checkpoint dir and W&B run name |
| `learning_rate` / `num_train_epochs` / `per_device_train_batch_size` | Standard training hyperparameters |
| `deepspeed` | DeepSpeed config — defaults to `ds_z3_config.json` |

> ⚠️ **`custom_moe_path` must be set before training.** Every YAML under [LLaMA-Factory/examples/train_full/qwen1.5moe/full/](LLaMA-Factory/examples/train_full/qwen1.5moe/full/) ships with this field blank:
>
> ```yaml
> custom_moe_path:             # path to the directory containing the custom MoE implementation
> ```
>
> If it is left empty, the ProbMoE block is **not** patched into the model and training silently falls back to the stock `Qwen2MoeSparseMoeBlock`. Fill it with the **absolute** path to `ProbMoE/models` (e.g. `/u/qgg5se/ProbMoE/models`) and confirm patching via the [log line below](#verifying-the-probmoe-patch).

**(b) In the launcher shell script** (e.g. [train_gsm.sh](LLaMA-Factory/run/qwen1.5/probmoe_full/train_gsm.sh)):

- Set `CUDA_VISIBLE_DEVICES` to your target GPUs (default `0,1,2,3`).
- The `CONFIG_FILES` array selects which YAML(s) to run — update if you renamed the config or want to sweep multiple LRs.

---

## 3. Launching a Training Run

The top-level launchers under [scripts/](scripts/) wrap the per-task scripts and handle cache/W&B environment variables.

### Fixed-k

```bash
cd train/qwen/scripts

# Edit:
#   - DATASET (gsm | law | summary | translation | code)
#   - WANDB_API_KEY
#   - the cd path to your local ProbMoE/train/qwen/LLaMA-Factory
bash train_ProbMoE_qwen_exact.sh
```

### Dynamic-k

```bash
cd train/qwen/scripts
# Edit:
#   - DATASET (gsm | law | summary | translation | code)
#   - WANDB_API_KEY
#   - the cd path to your local ProbMoE/train/qwen/LLaMA-Factory
bash "train_ProbMoE_qwen_dynamic.sh"
```

Both scripts read a single `DATASET` variable at the top, e.g.:

```bash
DATASET="gsm"            # one of: gsm, law, summary, translation, code
export WANDB_API_KEY=""  # paste your W&B key here
```

Logs are written to `LLaMA-Factory/logs/train_full_<config>.log`.

### Verifying the ProbMoE patch

If ProbMoE was patched into the model successfully, you should see a log line of the form:

```
[INFO|2026-05-10 17:12:00] llamafactory.model.loader:143 >> Found Qwen2MoeSparseMoeBlock module: model.layers.0.mlp (<class 'Qwen.ProbMoE_V1_qwen_exact.ProbMoEQwen2MoeSparseMoeBlock'>)
```
 If you don't see this, the ProbMoE module was not applied — check that `probmoe: true` is set in the YAML and that `custom_moe_path` resolves to a valid `ProbMoE/models` directory.

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
- **`custom_moe_path` errors:** the field must be an **absolute** path to the top-level `ProbMoE/models` directory in the YAML config.
- **OOM:** lower `per_device_train_batch_size`, switch to `ds_z3_config.json` (already the default), or reduce `cutoff_len`.
- **W&B not logging:** confirm `WANDB_API_KEY` is exported in `scripts/train_ProbMoE_exact_qwen.sh` before the launcher invokes `llamafactory-cli`.
- **Cache permission errors:** ensure `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, and `TRITON_CACHE_DIR` are set to writable paths in the top-level launcher.

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
