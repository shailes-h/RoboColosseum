#!/bin/bash
# Compute normalization stats with each repo's own tool (needed once per dataset before training).
# usage: bash scripts/data/norm_stats.sh <openpi|lingbot> [gpu_id=0]
#   openpi  -> $RC_OUTPUTS/norm_stats/openpi/pi05_yam/<TASK>_joint_v21/norm_stats.json   (~10 min with 16 workers)
#   lingbot -> $RC_OUTPUTS/norm_stats/lingbot/yam_<TASK>.json
# GR00T, MolmoAct2 and G0.5 compute their stats automatically at the start of training.
source "$(dirname "$0")/../../env.sh"
export CUDA_VISIBLE_DEVICES=${2:-0}
case "$1" in
  openpi)
    cd "$RC_TP/openpi"
    LD_LIBRARY_PATH=$RC_ENVS/ffmpeg7/lib:$LD_LIBRARY_PATH HF_LEROBOT_HOME=$RC_DATA/$EMBODIMENT RC_NORMSTATS_WORKERS=${WORKERS:-16} \
      "$RC_ENVS/openpi/bin/python" "$RC_ROOT/configs/openpi/run.py" scripts/compute_norm_stats.py --config-name pi05_yam;;
  lingbot)
    source "$RC_ROOT/scripts/train/lingbot_env.sh"
    mkdir -p "$RC_OUTPUTS/norm_stats/lingbot"
    bash train.sh scripts/compute_norm_stats.py ./configs/vla/norm_compute/post_data.yaml \
      --data.data_name yam --data.train_path "$(rc_ds joint)" \
      --data.robot_config_root "$RC_OUTPUTS/configs/lingbot/robot_configs" \
      --data.norm_path "$RC_OUTPUTS/norm_stats/lingbot/yam_${TASK}.json";;
  *) sed -n 2,6p "$0"; exit 1;;
esac
echo EXIT=$?
