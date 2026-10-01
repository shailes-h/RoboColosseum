#!/bin/bash
# Pre-download the base checkpoints into the Hugging Face cache (HF_HOME). Optional for GR00T,
# MolmoAct2 and pi0.5, which fetch their own on first use; required for G0.5 and LingBot.
# usage: bash scripts/setup/download_models.sh [gr00t|openpi|molmoact2|g05|lingbot|all]
source "$(dirname "$0")/../../env.sh"
HF=${HF_CLI:-hf}
case "${1:-all}" in
  gr00t)     $HF download nvidia/GR00T-N1.7-3B;;
  molmoact2) $HF download allenai/MolmoAct2;;
  g05)       $HF download OpenGalaxea/G05 --include "g05-base/*" "action_tokenizer.pt" "qwen3_5_2b_base_processor/*";;
  lingbot)   $HF download robbyant/lingbot-vla-v2-6b && $HF download Qwen/Qwen3-VL-4B-Instruct \
               && $HF download Ruicheng/moge-2-vitb-normal;;
  openpi)    # pi05_base lives on GCS; openpi caches it under $OPENPI_DATA_HOME (default ~/.cache/openpi) on first use.
             "$RC_ENVS/openpi/bin/python" -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/params'))";;
  all)       for m in gr00t molmoact2 g05 lingbot openpi; do bash "$0" $m || exit 1; done;;
  *)         sed -n 2,4p "$0"; exit 1;;
esac
