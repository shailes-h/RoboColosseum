#!/bin/bash
# Build the per-VLA copies of a BimanualYAM task from the raw LeRobot v3.0 dataset.
# usage: bash datasets/BimanualYAM/convert.sh [task]       (default task: $TASK = Dustpan)
# needs: the tools env (bash scripts/setup/install_env.sh tools). CPU only; ~20-40 min (mostly video re-encode).
#
#   <task>                 raw v3.0 from the Hub (EEF + joint columns)
#   <task>_joint           v3.0, joint space only           -> MolmoAct2, LingBot-VLA v2   (videos: symlink)
#   <task>_galaxea         v3.0, R1Lite-style split keys     -> G0.5                        (videos: symlink)
#   <task>_joint_v21       LeRobot v2.1 of _joint            -> pi0.5 (openpi)
#   <task>_joint_v21_gr00t v2.1, top cam letterboxed to 480x640 + meta/modality.json -> GR00T N1.7
set -e
source "$(dirname "$0")/../../env.sh"
EMBODIMENT=BimanualYAM; TASK=${1:-$TASK}
PY=$RC_ENVS/tools/bin/python; T=$RC_ROOT/scripts/data
D=$RC_DATA/$EMBODIMENT; RAW=$D/$TASK
[ -f "$RAW/meta/info.json" ] || { echo "missing $RAW; run datasets/$EMBODIMENT/download.sh first"; exit 1; }
[ -x "$PY" ] || { echo "missing tools env; run scripts/setup/install_env.sh tools"; exit 1; }

echo "== 1/4 ${TASK}_joint (joint-space v3.0)"
$PY $T/make_joint_dataset.py "$RAW" "$D/${TASK}_joint"

echo "== 2/4 ${TASK}_galaxea (R1Lite-style v3.0)"
$PY $T/make_galaxea_dataset.py "$D/${TASK}_joint" "$D/${TASK}_galaxea"

echo "== 3/4 ${TASK}_joint_v21 (LeRobot v2.1, GR00T's converter)"
STAGE=$D/_stage_v21; rm -rf "$STAGE" && mkdir -p "$STAGE"
cp -rL "$D/${TASK}_joint" "$STAGE/${TASK}_joint"   # converter wants real files, not the video symlink
(cd $RC_TP/Isaac-GR00T/scripts/lerobot_conversion && $PY convert_v3_to_v2.py --repo-id "${TASK}_joint" --root "$STAGE")
NEW=$(for d in "$STAGE"/${TASK}_joint*; do
        [ "$($PY -c "import json,sys;print(json.load(open(sys.argv[1]))['codebase_version'])" "$d/meta/info.json")" = v2.1 ] && echo "$d"
      done | head -1)
rm -rf "$D/${TASK}_joint_v21" && mv "$NEW" "$D/${TASK}_joint_v21" && rm -rf "$STAGE"

echo "== 4/4 ${TASK}_joint_v21_gr00t (GR00T stacks views, so all cameras must share one size)"
G=$D/${TASK}_joint_v21_gr00t; rm -rf "$G" && cp -a "$D/${TASK}_joint_v21" "$G"
$PY $T/pad_video.py "$G/videos/chunk-000/observation.images.top"    # 360x640 -> 480x640
$PY - "$G/meta/info.json" <<'PY'
import json, sys
p = sys.argv[1]; i = json.load(open(p)); f = i["features"]["observation.images.top"]
f["shape"] = [480, 640, 3]; f.setdefault("info", {})["video.height"] = 480
json.dump(i, open(p, "w"), indent=4)
PY
cp $RC_ROOT/configs/gr00t/yam_modality.json "$G/meta/modality.json"

du -shL "$D"/${TASK}*
echo DONE
