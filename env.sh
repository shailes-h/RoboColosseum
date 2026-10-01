# RoboColosseum paths and settings. Sourced by every script in scripts/.
# Override any of these in env.local.sh (untracked) for your cluster, e.g. fast node-local disks.
RC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$RC_ROOT/env.local.sh" ] && source "$RC_ROOT/env.local.sh"

export RC_ROOT
export RC_TP=$RC_ROOT/third_party                     # upstream VLA repos (git submodules, never edited)
export RC_ENVS=${RC_ENVS:-$RC_ROOT/.envs}             # one python env per VLA (keep on a fast local disk)
export RC_DATA=${RC_DATA:-$RC_ROOT/datasets}          # datasets root: $RC_DATA/<embodiment>/<dataset>
export RC_OUTPUTS=${RC_OUTPUTS:-$RC_ROOT/outputs}     # checkpoints, norm stats, generated configs
export RC_CACHE=${RC_CACHE:-$RC_ENVS/cache}           # uv / pip / conda / triton caches

export EMBODIMENT=${EMBODIMENT:-BimanualYAM}
export TASK=${TASK:-Dustpan}
export WANDB_PROJECT=${WANDB_PROJECT:-RoboColosseum}
# export WANDB_ENTITY=...                             # set in env.local.sh

export UV_CACHE_DIR=${UV_CACHE_DIR:-$RC_CACHE/uv} UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$RC_CACHE/uv_python} UV_LINK_MODE=copy
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$RC_CACHE/pip} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$RC_CACHE/triton}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$RC_CACHE/conda_pkgs}
# numpy madvise(MADV_HUGEPAGE) on large host buffers can trigger failing THP direct compaction on
# fragmented-memory nodes; checkpoint saves (openpi/orbax) then stall for hours. Harmless to disable.
export NUMPY_MADVISE_HUGEPAGE=0

rc_ds() { echo "$RC_DATA/$EMBODIMENT/$TASK${1:+_$1}"; }   # dataset dir; rc_ds joint -> .../Dustpan_joint
rc_ngpus() { echo "$1" | tr ',' '\n' | wc -l; }           # "0,1,2,3" -> 4
rc_py() { python3 -c "import math; print($1)"; }          # tiny arithmetic helper
