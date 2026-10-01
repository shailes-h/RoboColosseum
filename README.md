# RoboColosseum

A common harness for fine-tuning and comparing vision-language-action (VLA) models on the same robot data.
Each VLA is trained by its **own upstream repo, unmodified**: the repos are pinned git submodules in `third_party/`.
This repo adds only the glue on top: dataset conversion, per-VLA configs registered from outside the repos, env setup and
launch scripts.

| VLA | Upstream (submodule) | Base checkpoint | Framework |
|---|---|---|---|
| GR00T N1.7 | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `nvidia/GR00T-N1.7-3B` | PyTorch |
| G0.5 | [OpenGalaxea/GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) | `OpenGalaxea/G05` (g05-base) | PyTorch, hydra |
| MolmoAct2 | [allenai/molmoact2](https://github.com/allenai/molmoact2) | `allenai/MolmoAct2` | PyTorch |
| π0.5 | [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `gs://openpi-assets/checkpoints/pi05_base` | JAX |
| LingBot-VLA v2 6B | [robbyant/lingbot-vla-v2](https://github.com/robbyant/lingbot-vla-v2) | `robbyant/lingbot-vla-v2-6b` | PyTorch, conda |

Fine-tuned checkpoints: [`RoboColosseum/BimanualYAM-models`](https://huggingface.co/RoboColosseum/BimanualYAM-models).

## Layout

```
env.sh                 paths + settings sourced by every script (override in env.local.sh, gitignored)
third_party/           the 5 upstream repos (submodules, never edited)
datasets/<Embodiment>/ README + download.sh + convert.sh per embodiment; data itself is gitignored
configs/               our per-VLA configs, injected without touching the repos:
  gr00t/                 modality config (--modality-config-path) + modality.json
  openpi/                TrainConfig pi05_yam, registered at runtime by run.py
  molmoact2/             mixture "yam", registered at runtime by train.py
  galaxea/               hydra task/data yamls, added via hydra.searchpath
  lingbot/               train + robot config templates, rendered into outputs/ with local paths
scripts/
  setup/                 install_env.sh (one env per VLA), download_models.sh
  data/                  dataset converters, norm_stats.sh, stage.sh (copy data to a local disk)
  train/                 one launcher per VLA
  export/ eval/          MolmoAct2 -> HF conversion, GR00T open-loop eval
outputs/               checkpoints, norm stats, rendered configs (gitignored)
```

## Quick start

Requirements: Linux, NVIDIA GPUs with CUDA 12 drivers (we used 4x H100 80GB), [uv](https://docs.astral.sh/uv/),
conda (LingBot only), the `hf` CLI (`pip install -U huggingface_hub`), and logins: `hf auth login`, `wandb login`.

```bash
git clone --recursive <this repo> RoboColosseum && cd RoboColosseum
# (already cloned?)  git submodule update --init

cp env.local.sh.example env.local.sh       # optional: fast local disks, W&B entity, conda path, flash-attn wheel

# 1) environments: one per VLA, under $RC_ENVS. Run on a GPU node (same glibc/CUDA as training).
bash scripts/setup/install_env.sh all      # or: tools | gr00t | openpi | molmoact2 | g05 | lingbot | ffmpeg7
bash scripts/setup/download_models.sh all

# 2) data: download + convert (see datasets/BimanualYAM/README.md)
bash datasets/BimanualYAM/download.sh
bash datasets/BimanualYAM/convert.sh

# 3) norm stats (openpi and LingBot need them precomputed; the others compute their own)
bash scripts/data/norm_stats.sh openpi 0
bash scripts/data/norm_stats.sh lingbot 0

# 4) train:  <gpu_ids> <per_gpu_batch>
bash scripts/train/gr00t.sh     0,1,2,3 16
bash scripts/train/pi05.sh      0,1,2,3 16
bash scripts/train/molmoact2.sh 0,1,2,3 32
bash scripts/train/g05.sh       0,1,2,3 8
bash scripts/train/lingbot.sh   0,1,2,3 16
```

Each training script prints its run name, effective batch, LR and step count first, then writes to
`$RC_OUTPUTS/<model>/<run>/` and logs to W&B project `$WANDB_PROJECT`. Scripts run in the foreground; wrap them in
your scheduler (e.g. `sbatch --wrap` or `srun`) as needed. Common env knobs: `EPOCHS` (default 5), `RUN_SUFFIX`, `RUN`,
`MASTER_PORT`; per-model knobs are documented in each script's header.

## Training protocol

The aim is a fair comparison, so every model gets the same treatment:
- **Recipe:** each repo's own fine-tuning recipe (optimizer, schedule, frozen parts) from its closest example config.
- **LR scaling:** linear in the effective batch size relative to that recipe's batch size.
- **Duration:** 5 epochs, with one checkpoint per epoch.
- **Space:** joint space, 14-D absolute joint angles for state and action.
- **Precision:** fp32 master weights, bf16 compute.
- **Inputs:** 3 cameras plus the task instruction.

| VLA | Recipe source | GBS used | Peak LR (scaled) | Notes |
|---|---|---|---|---|
| GR00T N1.7 | `launch_finetune` defaults (GBS 64, 1e-4, cosine, 5% warmup) | 64 | 1e-4 | `FULL_VLM=1` (default here) also tunes LLM + vision. Keep GBS ≈ 64: at GBS 512 the run gets too few optimizer steps and stays on the constant-prediction plateau |
| G0.5 | R1Lite post-training (8/GPU, 4e-5, cosine, warmup 200) | 32 | 4e-5 | FSDP; vision LR x0.1 |
| MolmoAct2 | experiments README "Full Fine-Tuning" (GBS 64) | 128 | LLM 2e-5, ViT 1e-5, connector 1e-5, action expert 1e-4 | |
| π0.5 | `pi05_aloha_pen_uncap` (GBS 64, 2.5e-5 cosine, EMA 0.99) | 64 | 2.5e-5 | `adapt_to_pi=False` (YAM is not Aloha) |
| LingBot-VLA v2 | `configs/vla/real_robot/real_robot.yaml` (GBS 256, 5e-5 const, Muon) | 64 | 1.25e-5 | resumes automatically from the latest checkpoint |

## How the repos are kept untouched

| VLA | Mechanism |
|---|---|
| GR00T | repo CLI flags: `--modality-config-path configs/gr00t/yam_config.py`; `modality.json` is written into the dataset copy |
| π0.5 | `configs/openpi/run.py` registers `TrainConfig("pi05_yam")` into openpi's config table, then runs `scripts/train.py` unchanged. Norm stats go to `$RC_OUTPUTS` through `assets_base_dir` |
| MolmoAct2 | `configs/molmoact2/train.py` adds mixture `yam` to `MOLMOACT2_LEROBOT_MIXTURES`, then runs `launch_scripts/train_lerobot.py` unchanged |
| G0.5 | `hydra.searchpath=[file://configs/galaxea]` adds task/data `yam`; base-checkpoint paths are passed as hydra overrides |
| LingBot | config + robot config are passed as paths (`robot_config_root`, `--data.norm_stats_file`) |

`install_env.sh g05` restores GalaxeaVLA's `uv.lock` after `uv sync` rewrites it. Check that the submodules are still
clean with `git submodule foreach git status --short`.

## Adding things

- **New task (same embodiment):** run `download.sh <Task>` and `convert.sh <Task>`, then train with `TASK=<Task>` set.
- **New embodiment:** add `datasets/<Embodiment>/` (README, download.sh, convert.sh) and per-VLA configs for its
  state/action/camera layout in `configs/`. Then train with `EMBODIMENT=<Embodiment> TASK=<Task>`.
- **Bumping a VLA:** `cd third_party/<repo> && git checkout <commit>`, commit the submodule pointer, and re-run its smoke
  test (e.g. `EPOCHS=0.01`).

## Practical notes

- **Shared filesystems:** keep envs (`RC_ENVS`) and a staged data copy (`RC_DATA`) on node-local disk. On NFS, venv
  installs and per-sample video decoding are very slow.
- **Checkpoint stalls:** `env.sh` sets `NUMPY_MADVISE_HUGEPAGE=0`. Without it, the openpi/orbax checkpoint save could
  stall for hours on nodes with fragmented memory, because numpy hugepage advice triggers failing THP compaction.
- **flash-attn for LingBot:** a pip build needs a wheel that matches your glibc/CUDA/torch. Point `FLASH_ATTN_WHEEL` at
  a prebuilt wheel if the default install fails.
