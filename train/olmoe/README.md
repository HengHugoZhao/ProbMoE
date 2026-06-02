# ProbMoE Finetuning for OLMoE

This folder contains the finetuning pipeline for **ProbMoE** (Sparse-Interpolated Mixture of Experts) on top of the **OLMoE** base model. It is built on top of [open-instruct](open_instruct/) and supports multiple downstream tasks (GSM, Law, Summary, Translation, CodeAlpaca) with both fixed-`k` (*exact*) and dynamic-`k` (*band-k*) expert routing.

---

## Directory Layout

```
train/olmoe/
├── open_instruct/            # Modified open-instruct training 
│   ├── ds_configs/  
│   ├── finetune/             # Finetuning entry points and 
│   ├── run/                  # Per-task launch scripts
│   │   ├── gsm/
│   │   ├── law/
│   │   ├── summary/
│   │   ├── translation/
│   │   └── codealpaca/
│   ├── init_env.sh           # Environment bootstrap
│   └── requirements.txt
└── script/
    ├── train_ProbMoE_exact_OLMoE.sh         # Fixed-k entry point
    └── train_ProbMoE_exact_OLMoE_dynamic.sh # Dynamic-k entry point
```

---

## 1. Environment Setup

ProbMoE training requires Python 3.12, CUDA 12.1, and a recent PyTorch + Flash-Attention build.

```bash
# Create and activate the conda environment
conda create -n openinstruct python=3.12 -y
conda activate openinstruct

# Install the matching CUDA toolchain
conda install -c conda-forge cuda-nvcc=12.1 -y

# Install Python dependencies 
cd train/olmoe/open_instruct
bash init_env.sh
```

`init_env.sh` performs the following:
- Installs `torch==2.5.1` / `torchvision==0.20.1` against CUDA 12.1
- Builds `flash-attn==2.7.2.post1`
- Installs the rest of `requirements.txt`
- Installs the local `open_instruct` package in editable mode
- Downloads NLTK `punkt` tokenizer

---

## 2. Per-Task Configuration

Each supported dataset has its own folder under [open_instruct/run/](open_instruct/run/) containing two scripts:

| Script | Routing Mode | Purpose |
|---|---|---|
| `train_<dataset>.sh` | Fixed `k` | Train with a constant number of active experts |
| `train_<dataset>_dynamic.sh` | Band `k` | Train with `k` sampled in `[min_k, max_k]` per step |

Before launching a job, edit the dataset script (e.g. [open_instruct/run/gsm/train_gsm.sh](open_instruct/run/gsm/train_gsm.sh)) to match your hardware and paths:

| Flag | Description |
|---|---|
| `--num_gpus` | Number of GPUs to use |
| `--devices` | CUDA visible devices, e.g. `0,1,2,3` |
| `--port` | Master port for distributed training |
| `--total_batch_size` | Global batch size (gradient accumulation derived automatically) |
| `--per_device_train_batch_size` / `--per_device_eval_batch_size` | Per-GPU batch size |
| `--lr` / `--lr_scheduler_type` | Learning rate and schedule |
| `--num_train_epochs` | Training epochs |
| `--gradient_checkpointing` | Trade compute for memory |
| `--probmoe` | Enable ProbMoE routing (set `True`) |
| `--custom_moe_path` | **Absolute** path to `ProbMoE/models` (e.g. `/u/qgg5se/ProbMoE/models`) — ⚠️ **must be set to your local path before training** |
| `--v2` | Use the faster ProbMoE kernel (set `True` for the optimized path) |
| `--band_k` | `True` for dynamic-k, `False` for fixed-k |
| `--max_k` / `--min_k` | Upper / lower bound on active experts |
| `--freeze_gate` | Freeze the router during finetuning |
| `--seed` | Random seed |
| `--output_suffix` | Tag appended to the run name and checkpoint dir |

> ⚠️ **`--custom_moe_path` must be set before training.** Each per-task script under [open_instruct/run/](open_instruct/run/) ships with a placeholder absolute path:
>
> ```bash
> --custom_moe_path "/u/qgg5se/ProbMoE/models" \   # path to the directory containing the custom MoE implementation
> ```
>
> Update it to the **absolute** path of `ProbMoE/models` on your machine. If it is missing or wrong, the ProbMoE block is **not** patched into the model and training silently falls back to the stock `OlmoeSparseMoeBlock`. Confirm patching via the [log line below](#verifying-the-probmoe-patch).

> **Note:** to switch to the faster ProbMoE implementation, set `--v2 True` in the per-task script. To reproduce the exact number in the paper, use V1.

---

## 3. Launching a Training Run

The top-level launchers under [script/](script/) wrap the per-task scripts and route logs into `log/`.

### Fixed-k

```bash
cd train/olmoe/script

# Edit DATASET (gsm | law | summary | translation | codealpaca)
# Set WANDB_API_KEY
# Update the cd path to your local ProbMoE/train/olmoe/open_instruct
bash train_ProbMoE_OLMoE_exact.sh
```

### Dynamic-k

```bash
cd train/olmoe/script
# Edit DATASET (gsm | law | summary | translation | codealpaca)
# Set WANDB_API_KEY
# Update the cd path to your local ProbMoE/train/olmoe/open_instruct
bash train_ProbMoE_OLMoE_dynamic.sh
```

Both scripts read a single `DATASET` variable at the top, e.g.:
Logs are written to `log/exact/<dataset>/` and `log/dynamic/<dataset>/` respectively.

### Verifying the ProbMoE patch

If ProbMoE was patched into the model successfully, you should see a log line of the form:

```
[RANK 0] Found MoE block: model.layers.0.mlp - Type: <class 'OLMoE.v1.ProbMoE_V1_olmoe_dynamic.ProbMoEOlmoeSparseMoeBlock'>
```

If you don't see this, the ProbMoE module was not applied — check that `--probmoE True` is set and that `--custom_moe_path` resolves to a valid `ProbMoE/models` directory.

---

## 4. Supported Datasets

| Dataset | Folder | Task |
|---|---|---|
| `gsm` | [run/gsm](open_instruct/run/gsm/) | Grade-school math reasoning |
| `law` | [run/law](open_instruct/run/law/) | Legal-domain instruction following |
| `summary` | [run/summary](open_instruct/run/summary/) | Summarization |
| `translation` | [run/translation](open_instruct/run/translation/) | Machine translation |
| `codealpaca` | [run/codealpaca](open_instruct/run/codealpaca/) | Code instruction following |

---

## 5. Evaluation

Evaluation lives in a separate folder. See [eval/README.md](../../eval/README.md) for task-specific evaluation instructions (GSM, MBPP, ESFT benchmarks).

---

## Troubleshooting

- **`flash-attn` build fails:** ensure `cuda-nvcc=12.1` is on the path and `packaging` is installed before `flash-attn`.
- **`custom_moe_path` errors:** the path must be **absolute** and point to the top-level `ProbMoE/models` directory.
- **OOM:** reduce `--per_device_train_batch_size`, enable `--gradient_checkpointing True`, or lower `--max_k`.
- **W&B not logging:** confirm `WANDB_API_KEY` is exported in the launcher script before `bash run/...`.

---

## Citations 

Following models, codebases, and datasets are used:

- **OLMoE-1B-7B** — base model used for finetuning.
  [allenai/OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
- **open-instruct** — finetuning framework adapted for ProbMoE training.
  [allenai/open-instruct](https://github.com/allenai/open-instruct)
- **ESFT** — project and datasets (Law, Summary, Translation) used for domain-specific finetuning and evaluation.
  [deepseek-ai/ESFT](https://github.com/deepseek-ai/ESFT)
- **GSM8K** — grade-school math reasoning benchmark (Cobbe et al., 2021).
  [openai/grade-school-math](https://github.com/openai/grade-school-math) · [Paper](https://arxiv.org/abs/2110.14168)
