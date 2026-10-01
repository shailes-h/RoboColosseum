#!/bin/bash
# GR00T open-loop eval (repo tool): predicted vs. ground-truth actions on a few training episodes, saved as plots.
# usage: bash scripts/eval/gr00t_open_loop.sh <gpu_id> <checkpoint_dir> <out_dir>
source "$(dirname "$0")/../../env.sh"
[ -n "$3" ] || { sed -n 2,3p "$0"; exit 1; }
mkdir -p "$3"
cd "$RC_TP/Isaac-GR00T"
CUDA_VISIBLE_DEVICES=$1 "$RC_ENVS/gr00t/bin/python" gr00t/eval/open_loop_eval.py \
  --dataset-path "$(rc_ds joint_v21_gr00t)" --embodiment-tag NEW_EMBODIMENT \
  --model-path "$2" --traj-ids 0 25 50 75 100 --execution-horizon 16 --steps 400 \
  --save-plot-path "$3"
echo EXIT=$?
