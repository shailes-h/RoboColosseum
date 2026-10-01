#!/bin/bash
# Convert a MolmoAct2 training checkpoint (FSDP shards, e.g. outputs/molmoact2/<run>/step861) to a Hugging Face
# model folder (safetensors + processor + norm_stats.json), with the repo's own converter.
# usage: bash scripts/export/molmoact2_to_hf.sh <checkpoint_dir> <out_dir>
source "$(dirname "$0")/../../env.sh"
CK=$(realpath "$1"); OUT=$(realpath -m "$2"); [ -n "$2" ] || { sed -n 2,4p "$0"; exit 1; }
cd "$RC_TP/molmoact2/experiments"
"$RC_ENVS/molmoact2/bin/python" olmo/hf_model/convert_molmoact2_to_hf.py "$CK" "$OUT"
echo EXIT=$?
