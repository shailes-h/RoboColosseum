#!/bin/bash
# MolmoAct2 (allenai/MolmoAct2 base) full fine-tune on <EMBODIMENT>/<TASK>_joint. Mixture "yam" (configs/molmoact2/train.py).
# usage: bash scripts/train/molmoact2.sh <gpu_ids> [per_gpu_bs=32]   e.g.  bash scripts/train/molmoact2.sh 0,1,2,3 32
#   env: EPOCHS=5  RUN_SUFFIX=  RUN=
# Recipe (experiments/README.md "Full Fine-Tuning"): GBS 64; LR llm 1e-5, vit 5e-6, connector 5e-6, action expert 5e-5;
# warmup 200; decay to 0.1x. All LRs scaled linearly with GBS. Checkpoints are FSDP shards, one per epoch;
# convert with scripts/export/molmoact2_to_hf.sh.
source "$(dirname "$0")/common.sh" molmoact2 "$1" "${2:-32}"
echo "RUN=$RUN GBS=$GBS STEPS=$STEPS SAVE_EVERY=$SAVE"
FF=$RC_ENVS/ffmpeg7/lib; NPP=$(ls -d "$RC_ENVS"/molmoact2/lib/python3*/site-packages/nvidia/npp/lib)
export LEROBOT_DATA_ROOT=$RC_DATA MOLMO_DATA_DIR=$RC_CACHE/molmo_data LD_LIBRARY_PATH=$NPP:$FF:$LD_LIBRARY_PATH  # torchcodec needs FFmpeg 7
export HF_ACCESS_TOKEN=${HF_ACCESS_TOKEN:-$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token" 2>/dev/null)}
export WANDB_API_KEY=${WANDB_API_KEY:-$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' ~/.netrc 2>/dev/null)}
mkdir -p "$MOLMO_DATA_DIR"
cd "$RC_TP/molmoact2/experiments"
"$RC_ENVS/molmoact2/bin/torchrun" --standalone --nproc-per-node=$NGPU "$RC_ROOT/configs/molmoact2/train.py" \
  allenai/MolmoAct2 yam \
  --wandb.name="$RUN" ${WANDB_ENTITY:+--wandb.entity=$WANDB_ENTITY} --wandb.project="$WANDB_PROJECT" \
  --max_duration=$STEPS --device_batch_size=$BS --global_batch_size=$GBS \
  --num_workers=8 --pin_memory=true --data.timeout=900 \
  --save_interval=$SAVE --save_num_checkpoints_to_keep=$KEEP --save_folder="$OUT" \
  --packing=false --dynamic_seq_len=true \
  --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head --lora_enable=false \
  --llm_learning_rate=$(scale 1e-5 64) --vit_learning_rate=$(scale 5e-6 64) \
  --connector_learning_rate=$(scale 5e-6 64) --action_expert_learning_rate=$(scale 5e-5 64)
echo EXIT=$?
