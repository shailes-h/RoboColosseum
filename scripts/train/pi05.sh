#!/bin/bash
# pi0.5 (openpi JAX, pi05_base) full fine-tune on <EMBODIMENT>/<TASK>_joint_v21. Config pi05_yam (configs/openpi/yam.py).
# usage: bash scripts/train/pi05.sh <gpu_ids> <per_gpu_bs> [extra trainer args...]         e.g.  bash scripts/train/pi05.sh 0,1,2,3 16
#   env: EPOCHS=5  FSDP=<n_gpus>  RUN_SUFFIX=  RUN=
# Needs norm stats first: bash scripts/data/norm_stats.sh openpi <gpu>
# Recipe (pi05_aloha_pen_uncap): GBS 64, cosine peak 2.5e-5 -> 2.5e-6, warmup 1k, EMA 0.99. LR scaled linearly with
# GBS; cosine decays over the whole run. One checkpoint per epoch, all kept.
source "$(dirname "$0")/common.sh" pi05 "$1" "$2"
LR=$(scale 2.5e-5 64)
echo "RUN=$RUN GBS=$GBS LR=$LR STEPS=$STEPS SAVE_EVERY=$SAVE"
cd "$RC_TP/openpi"
export HF_LEROBOT_HOME=$RC_DATA/$EMBODIMENT XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
"$RC_ENVS/openpi/bin/python" "$RC_ROOT/configs/openpi/run.py" scripts/train.py pi05_yam \
  --exp-name "$RUN" --project-name "$WANDB_PROJECT" --overwrite \
  --checkpoint-base-dir "$RC_OUTPUTS/pi05" \
  --batch-size $GBS --num-train-steps $STEPS --fsdp-devices ${FSDP:-$NGPU} --num-workers 8 \
  --lr-schedule.peak-lr $LR --lr-schedule.decay-lr $(rc_py "$LR / 10") --lr-schedule.decay-steps $STEPS \
  --save-interval $SAVE --keep-period $SAVE "${@:3}"
echo EXIT=$?
