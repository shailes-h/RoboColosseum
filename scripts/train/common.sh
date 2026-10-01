# Shared by scripts/train/*.sh:  source "$(dirname "$0")/common.sh" <model> <gpu_ids> <per_gpu_bs>
# Sets GPUS NGPU BS GBS(=BS*NGPU*ACC) FRAMES EPOCHS STEPS SAVE(=1 epoch) KEEP RUN OUT, and scale() for linear LR scaling.
source "$(dirname "${BASH_SOURCE[0]}")/../../env.sh"
MODEL=$1; GPUS=$2; BS=$3
[ -n "$BS" ] || { sed -n '2,/^set\|^source/p' "$0" | grep '^#'; exit 1; }
NGPU=$(rc_ngpus "$GPUS"); ACC=${ACC:-1}; GBS=$((BS * NGPU * ACC)); EPOCHS=${EPOCHS:-5}
INFO=$RC_DATA/$EMBODIMENT/$TASK/meta/info.json
FRAMES=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_frames'])" "$INFO") || { echo "missing $INFO"; exit 1; }
STEPS=$(rc_py "math.ceil($EPOCHS * $FRAMES / $GBS)"); SAVE=$(rc_py "math.ceil($FRAMES / $GBS)")
KEEP=$(rc_py "math.ceil($EPOCHS) + 1")   # checkpoints to keep (one per epoch, plus slack)
RUN=${RUN:-$MODEL-$(echo "$EMBODIMENT-$TASK" | tr 'A-Z' 'a-z')-gbs$GBS${RUN_SUFFIX}}
OUT=$RC_OUTPUTS/$MODEL/$RUN
scale() { rc_py "$1 * $GBS / $2"; }   # scale <recipe_lr> <recipe_gbs>: linear LR scaling with the effective batch
export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p "$RC_OUTPUTS/$MODEL"
