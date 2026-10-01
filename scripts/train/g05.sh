#!/bin/bash
# G0.5 (OpenGalaxea/G05 g05-base) full fine-tune on <EMBODIMENT>/<TASK>_galaxea. Task "yam" (configs/galaxea/).
# usage: bash scripts/train/g05.sh <gpu_ids> [per_gpu_bs=8]        e.g.  bash scripts/train/g05.sh 0,1,2,3 8
#   env: EPOCHS=5  RUN_SUFFIX=  RUN=
# Recipe (R1Lite post-training): LR 4e-5 at 8/GPU, cosine, warmup 200, wd 0.03. Per-GPU batch = recipe default, so the
# LR is not scaled. FSDP (repo option) is needed to fit full fine-tuning on 80GB GPUs.
# Needs: bash scripts/setup/download_models.sh g05
source "$(dirname "$0")/common.sh" g05 "$1" "${2:-8}"
G05=$(${HF_CLI:-hf} download OpenGalaxea/G05 --include "g05-base/*" "action_tokenizer.pt" "qwen3_5_2b_base_processor/*" --quiet)
echo "RUN=$RUN GBS=$GBS (base: $G05)"
source "$RC_ENVS/g05/bin/activate"
cd "$RC_TP/GalaxeaVLA"
# Same environment as the repo's scripts/run/finetune.sh, which only accepts task yamls inside the repo;
# our task/data yamls are added through hydra.searchpath instead.
export HYDRA_FULL_ERROR=1 OC_CAUSE=1 HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$PWD:$PYTHONPATH G05_OUTPUT_DIR=$RC_OUTPUTS/g05 EXP_NAME=$RUN
python -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node $NGPU scripts/finetune.py task=yam \
  "hydra.searchpath=[file://$RC_ROOT/configs/galaxea]" \
  model.pretrained_ckpt=$G05/g05-base/checkpoints/model_state_dict.pt \
  model.model_arch.hf_processor_path=$G05/qwen3_5_2b_base_processor \
  tokenizer.vq_config.ckpt_dir=$G05/action_tokenizer.pt \
  use_fsdp=true model.batch_size=$BS model.max_epochs=$EPOCHS \
  logger.mode=online logger.project=$WANDB_PROJECT ${WANDB_ENTITY:+logger.workspace=$WANDB_ENTITY} logger.experiment_name=$RUN
echo EXIT=$?
