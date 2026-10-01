#!/bin/bash
# GR00T N1.7 (nvidia/GR00T-N1.7-3B) fine-tune on <EMBODIMENT>/<TASK>_joint_v21_gr00t.
# usage: bash scripts/train/gr00t.sh <gpu_ids> <per_gpu_bs> [extra trainer args...]        e.g.  bash scripts/train/gr00t.sh 0,1,2,3 16
#   env: EPOCHS=5  ACC=1 (grad accumulation)  FULL_VLM=1 (also tune LLM + vision; 0 = repo default: action head only)
#        LR_MULT=1  RUN_SUFFIX=  RUN=
# Recipe (launch_finetune defaults): GBS 64 @ LR 1e-4, cosine, 5% warmup. LR scaled linearly with GBS (x LR_MULT).
# GBS 64 matters: at GBS 512 the run gets too few optimizer steps (~540 for 5 epochs) and never leaves the
# constant-prediction loss plateau. One checkpoint per epoch, all kept.
source "$(dirname "$0")/common.sh" gr00t "$1" "$2"
LR=$(rc_py "$(scale 1e-4 64) * ${LR_MULT:-1}"); TUNE=""
[ "${FULL_VLM:-1}" = 1 ] && TUNE="--tune-llm --tune-visual"
echo "RUN=$RUN GBS=$GBS LR=$LR STEPS=$STEPS SAVE_EVERY=$SAVE TUNE=$TUNE"
cd "$RC_TP/Isaac-GR00T"
"$RC_ENVS/gr00t/bin/torchrun" --nproc_per_node=$NGPU --master_port=${MASTER_PORT:-29610} gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path "$(rc_ds joint_v21_gr00t)" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$RC_ROOT/configs/gr00t/yam_config.py" \
  --num-gpus $NGPU --output-dir "$OUT" --experiment-name "$RUN" \
  --max-steps $STEPS --save-steps $SAVE --save-total-limit $KEEP \
  --global-batch-size $((BS * NGPU)) --gradient-accumulation-steps $ACC --learning-rate $LR --dataloader-num-workers 8 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --use-wandb --wandb-project "$WANDB_PROJECT" $TUNE "${@:3}"
echo EXIT=$?
