#!/bin/bash
# Build the per-VLA copies of a DROID task from the raw LeRobot v3.0 dataset (same layout as datasets/BimanualYAM).
# usage: bash datasets/DROID/convert.sh <task>
# needs: the tools env (bash scripts/setup/install_env.sh tools). CPU only.
#
#   <task>                 raw v3.0 from the Hub (joint state/action + cartesian, cameras left_image right_image wrist)
#   <task>_joint           v3.0, joint space only           -> MolmoAct2, LingBot-VLA v2   (videos: symlink)
#   <task>_galaxea         v3.0, R1Lite-style split keys     -> G0.5                        (videos: symlink)
#   <task>_joint_v21       LeRobot v2.1 of _joint            -> pi0.5 (openpi)
#   <task>_joint_v21_gr00t v2.1 + meta/modality.json         -> GR00T N1.7 (all cameras are 360x640, no padding)
set -e
source "$(dirname "$0")/../../env.sh"
EMBODIMENT=DROID; TASK=${1:-$TASK}
PY=$RC_ENVS/tools/bin/python; T=$RC_ROOT/scripts/data
D=$RC_DATA/$EMBODIMENT; RAW=$D/$TASK
[ -f "$RAW/meta/info.json" ] || { echo "missing $RAW; run datasets/$EMBODIMENT/download.sh first"; exit 1; }
[ -x "$PY" ] || { echo "missing tools env; run scripts/setup/install_env.sh tools"; exit 1; }

echo "== 0/4 check video metadata against the video files"
$PY $T/check_videos.py "$RAW" || { echo "video metadata does not match the video files; fix the dataset first"; exit 1; }

echo "== 1/4 ${TASK}_joint (joint-space v3.0)"
$PY $T/make_joint_dataset.py "$RAW" "$D/${TASK}_joint"

echo "== 2/4 ${TASK}_galaxea (R1Lite-style v3.0)"
$PY $T/make_galaxea_dataset.py "$D/${TASK}_joint" "$D/${TASK}_galaxea"

echo "== 3/4 ${TASK}_joint_v21 (LeRobot v2.1, GR00T's converter)"
STAGE=$D/_stage_v21_$TASK; rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -rL "$D/${TASK}_joint" "$STAGE/${TASK}_joint"   # converter wants real files, not the video symlink
(cd $RC_TP/Isaac-GR00T/scripts/lerobot_conversion && $PY convert_v3_to_v2.py --repo-id "${TASK}_joint" --root "$STAGE")
NEW=$(for d in "$STAGE"/${TASK}_joint*; do
        [ "$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['codebase_version'])" "$d/meta/info.json")" = v2.1 ] && echo "$d"
      done | head -1)
rm -rf "$D/${TASK}_joint_v21" && mv "$NEW" "$D/${TASK}_joint_v21" && rm -rf "$STAGE"

echo "== 4/4 ${TASK}_joint_v21_gr00t (v2.1 + GR00T modality)"
G=$D/${TASK}_joint_v21_gr00t; rm -rf "$G" && cp -a "$D/${TASK}_joint_v21" "$G"
cp $RC_ROOT/configs/gr00t/rc_droid_modality.json "$G/meta/modality.json"

du -shL "$D"/${TASK}*
echo DONE
