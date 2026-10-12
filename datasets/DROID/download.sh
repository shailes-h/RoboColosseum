#!/bin/bash
# Download a raw DROID (Franka) LeRobot v3.0 task dataset from the Hugging Face Hub.
# usage: bash datasets/DROID/download.sh <MugOnCoaster|CloseLaptop>      -> $RC_DATA/DROID/<task>/
set -e
source "$(dirname "$0")/../../env.sh"
EMBODIMENT=DROID; TASK=${1:-$TASK}
case $TASK in
  MugOnCoaster) HF_REPO=aarontung/droid-mug-on-coaster;;
  CloseLaptop)  HF_REPO=aarontung/droid-close-laptop;;
  *) echo "unknown DROID task $TASK"; exit 1;;
esac
mkdir -p "$RC_DATA/$EMBODIMENT"
${HF_CLI:-hf} download "$HF_REPO" --repo-type dataset --local-dir "$RC_DATA/$EMBODIMENT/$TASK"
python3 -c "import json,sys; i=json.load(open(sys.argv[1])); print('$TASK:', i['codebase_version'], i['total_episodes'], 'episodes', i['total_frames'], 'frames')" \
  "$RC_DATA/$EMBODIMENT/$TASK/meta/info.json"
