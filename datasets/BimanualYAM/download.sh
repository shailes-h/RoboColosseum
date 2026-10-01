#!/bin/bash
# Download the raw BimanualYAM LeRobot v3.0 dataset from the Hugging Face Hub.
# usage: bash datasets/BimanualYAM/download.sh [task]      (default task: $TASK = Dustpan)
#   -> $RC_DATA/BimanualYAM/<task>/
set -e
source "$(dirname "$0")/../../env.sh"
EMBODIMENT=BimanualYAM; TASK=${1:-$TASK}
HF_REPO=${HF_REPO:-RoboColosseum/BimanualYAM-datasets}
mkdir -p "$RC_DATA/$EMBODIMENT"
${HF_CLI:-hf} download "$HF_REPO" --repo-type dataset --include "$TASK/*" --local-dir "$RC_DATA/$EMBODIMENT"
python3 -c "import json,sys; i=json.load(open(sys.argv[1])); print('$TASK:', i['codebase_version'], i['total_episodes'], 'episodes', i['total_frames'], 'frames')" \
  "$RC_DATA/$EMBODIMENT/$TASK/meta/info.json"
