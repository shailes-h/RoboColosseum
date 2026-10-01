#!/bin/bash
# LingBot-VLA v2 6B (robbyant/lingbot-vla-v2-6b) full post-train on <EMBODIMENT>/<TASK>_joint. Config configs/lingbot/yam.yaml.
# usage: bash scripts/train/lingbot.sh <gpu_ids> [per_gpu_bs=16]   e.g.  bash scripts/train/lingbot.sh 0,1,2,3 16
#   env: EPOCHS=5  RUN_SUFFIX=  RUN=
# Needs: bash scripts/setup/download_models.sh lingbot; bash scripts/data/norm_stats.sh lingbot <gpu>
# Recipe (configs/vla/real_robot/real_robot.yaml): GBS 256 @ LR 5e-5 constant, Muon. LR scaled linearly with GBS.
# Precision: fp32 master weights, bf16 compute (FSDP2 mixed precision). Re-running the same RUN resumes from its
# latest checkpoint (enable_resume). One checkpoint per epoch.
source "$(dirname "$0")/common.sh" lingbot "$1" "${2:-16}"
LR=$(scale 5e-5 256)
echo "RUN=$RUN GBS=$GBS LR=$LR STEPS=$STEPS SAVE_EVERY=$SAVE"
source "$(dirname "$0")/lingbot_env.sh"
bash train.sh tasks/vla/train_lingbotvla.py "$CFG" \
  --data.norm_stats_file "$RC_OUTPUTS/norm_stats/lingbot/yam.json" \
  --train.output_dir "$OUT" --train.micro_batch_size $BS --train.global_batch_size $GBS \
  --train.lr $LR --train.max_steps $STEPS --train.save_steps $SAVE \
  --train.enable_fp32 false --train.enable_mixed_precision true \
  --train.use_wandb true --train.wandb_project "$WANDB_PROJECT" --train.wandb_name "$RUN"
echo EXIT=$?
