#!/bin/bash
# Copy datasets from shared storage to a fast node-local disk (training decodes video from every sample;
# NFS is far too slow). Point RC_DATA at the destination afterwards (env.local.sh).
# usage: bash scripts/data/stage.sh <src_datasets_root> <dst_datasets_root>
#   e.g. bash scripts/data/stage.sh /shared/RoboColosseum/datasets /tmp/rc_data
set -e
SRC=$1; DST=$2; [ -n "$DST" ] || { echo "usage: stage.sh <src> <dst>"; exit 1; }
mkdir -p "$DST"
cp -a "$SRC"/. "$DST"/     # -a keeps the relative video symlinks of the *_joint / *_galaxea copies
du -sh "$DST"
