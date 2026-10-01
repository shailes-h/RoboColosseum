#!/bin/bash
# Create one python env under $RC_ENVS/<name>, following each upstream repo's own install recipe.
# usage: bash scripts/setup/install_env.sh <tools|gr00t|openpi|molmoact2|g05|lingbot|ffmpeg7|all>
#   tools     dataset conversion (GR00T's LeRobot v3->v2.1 converter + pyarrow/av)    uv, python 3.11
#   gr00t     Isaac-GR00T (uv sync)                                                   uv, python 3.12
#   openpi    openpi / pi0.5 (uv sync, JAX)                                           uv
#   molmoact2 MolmoAct2 experiments + its bundled lerobot                             uv, python 3.12
#   g05       GalaxeaVLA / G0.5 (uv sync)                                             uv
#   lingbot   LingBot-VLA v2 (repo's tools/create_train_env.sh)                       conda; set CONDA_SH, FLASH_ATTN_WHEEL
#   ffmpeg7   FFmpeg 7 shared libs for torchcodec (MolmoAct2, LingBot)                conda
# Needs: uv (https://docs.astral.sh/uv), conda for lingbot/ffmpeg7, CUDA 12 driver. Run on a compute node
# with the target GPUs/glibc (envs are not portable across very different nodes).
set -o pipefail
source "$(dirname "$0")/../../env.sh"
NAME=$1; V=$RC_ENVS/$NAME
mkdir -p "$RC_ENVS" "$RC_CACHE"
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=8 MAX_JOBS=${MAX_JOBS:-16} CMAKE_POLICY_VERSION_MINIMUM=3.5 GIT_LFS_SKIP_SMUDGE=1
conda_init() { source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; }

case "$NAME" in
  tools)
    uv venv --clear --python 3.11 "$V" \
      && uv pip install --python "$V/bin/python" -e "$RC_TP/Isaac-GR00T/scripts/lerobot_conversion" av pyarrow numpy \
      && "$V/bin/python" -c "import av, pyarrow, lerobot; print('OK tools')";;
  gr00t)
    cd "$RC_TP/Isaac-GR00T" && UV_PROJECT_ENVIRONMENT=$V uv sync --python 3.12 \
      && "$V/bin/python" -c "import gr00t, torch; print('OK gr00t', torch.__version__)";;
  openpi)
    git -C "$RC_TP/openpi" submodule update --init --recursive \
      && cd "$RC_TP/openpi" && UV_PROJECT_ENVIRONMENT=$V uv sync \
      && uv pip install --python "$V/bin/python" -e . \
      && "$V/bin/python" -c "import openpi, jax; print('OK openpi', jax.__version__)";;
  molmoact2)
    git -C "$RC_TP/molmoact2" submodule update --init --recursive \
      && cd "$RC_TP/molmoact2/experiments" && uv venv --clear --python 3.12 "$V" \
      && uv pip install --python "$V/bin/python" -e ".[all]" \
      && uv pip install --index-strategy unsafe-best-match --python "$V/bin/python" -e "./lerobot[async]" debugpy "protobuf==6.33.5" "nvidia-npp-cu12>=12.4" \
      && "$V/bin/python" -c "import torch, olmo; print('OK molmoact2', torch.__version__)";;
  g05)
    # `uv sync` rewrites the repo's uv.lock; restore it so the submodule stays pristine.
    cd "$RC_TP/GalaxeaVLA" && UV_PROJECT_ENVIRONMENT=$V uv sync --index-strategy unsafe-best-match; rc=$?
    git -C "$RC_TP/GalaxeaVLA" checkout -- uv.lock
    [ $rc = 0 ] && "$V/bin/python" -c "import g05, torch; print('OK g05', torch.__version__)";;
  lingbot)
    conda_init && export CONDA_ENVS_PATH=$RC_ENVS \
      && cd "$RC_TP/lingbot-vla-v2" && bash tools/create_train_env.sh --env-name lingbot \
      && conda activate "$V" && pip install "nvidia-npp-cu12>=12.4" \
      && python -c "import torch, flash_attn, flash_attn_2_cuda, lingbotvla; print('OK lingbot', torch.__version__)";;
  ffmpeg7)
    conda_init && conda create -y -p "$V" -c conda-forge "ffmpeg=7.1.1=gpl*" && ls "$V/lib/libavutil.so.59";;
  all)
    for n in tools gr00t openpi molmoact2 g05 ffmpeg7 lingbot; do bash "$0" $n > "$RC_ENVS/install_$n.log" 2>&1 & done
    wait; grep -H "^EXIT=" "$RC_ENVS"/install_*.log; exit;;
  *) sed -n 2,13p "$0"; exit 1;;
esac
echo EXIT=$?
