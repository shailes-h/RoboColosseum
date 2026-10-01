# Sourced by lingbot.sh and scripts/data/norm_stats.sh: activates the env, resolves base-model paths, and renders the
# config templates (configs/lingbot/) into $RC_OUTPUTS/configs/lingbot/. Sets CFG; leaves cwd at the repo root.
source "${CONDA_SH:-$(conda info --base)/etc/profile.d/conda.sh}"; conda activate "$RC_ENVS/lingbot"
HF=${HF_CLI:-hf}
export LINGBOT_DIR=$($HF download robbyant/lingbot-vla-v2-6b --quiet) QWEN3VL_DIR=$($HF download Qwen/Qwen3-VL-4B-Instruct --quiet) \
       MOGE_DIR=$($HF download Ruicheng/moge-2-vitb-normal --quiet)
G=$RC_OUTPUTS/configs/lingbot; mkdir -p "$G/robot_configs"
render() { python3 -c "import os,sys; open(sys.argv[2],'w').write(os.path.expandvars(open(sys.argv[1]).read()))" "$1" "$2"; }
render "$RC_ROOT/configs/lingbot/yam.yaml" "$G/yam.yaml"
render "$RC_ROOT/configs/lingbot/robot_configs/yam.yaml" "$G/robot_configs/yam.yaml"
CFG=$G/yam.yaml
NPP=$(ls -d "$RC_ENVS"/lingbot/lib/python3*/site-packages/nvidia/npp/lib)
export LD_LIBRARY_PATH=$NPP:$RC_ENVS/ffmpeg7/lib:$LD_LIBRARY_PATH   # torchcodec needs FFmpeg 7
export MASTER_PORT=${MASTER_PORT:-62610} WANDB_NAME=${RUN:-lingbot}
cd "$RC_TP/lingbot-vla-v2"
