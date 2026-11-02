#!/bin/bash
#SBATCH --job-name=pf_reference
#SBATCH --qos=qos_gpu-dev            # 2 h cap, starts fast
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10           # 1/4 node = billed as 1 GPU
#SBATCH --hint=nomultithread
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Research reference model (M0) on Jean Zay, one V100, in 2 h links. Each link runs the first
# unfinished step and exits: Backbone training, Versatile training, evaluation (centered cameras),
# evaluation (wild cameras) of the Versatile model, then of the Backbone. VARIANT=scratch trains the Versatile model from scratch instead (no
# Backbone), in its own folder. VARIANT=full trains it from scratch on the
# v1 Train clips plus the extra Train clips (full AMASS, FULL_EPOCHS epochs, default 6) and also evaluates it on
# the Unseen clips. Training stops itself before the walltime and resumes mid-epoch, so the
# same command is submitted again and again, for example as a chain of links (--dependency=afterany).
# The project account is site-specific: pass it on the command line.
#   sbatch --account=<account> slurm/jz/reference_dev.slurm
# DATA and RUNS override the default folders; TRAIN_ARGS and EVAL_ARGS add flags (local checks).
set -eu

set +u                               # module files reference unset variables
module purge
module load pytorch-gpu/py3/2.8.0
source $WORK/venvs/priorformer/bin/activate
set -u
export HF_HOME=$WORK/.cache/huggingface TORCH_HOME=$WORK/.cache/torch \
       XDG_CACHE_HOME=$WORK/.cache MPLCONFIGDIR=$WORK/.cache/matplotlib
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1

REPO=$WORK/PriorFormer
DATA=${DATA:-$SCRATCH/priorformer/data/amass_shaped}
RUNS=${RUNS:-$SCRATCH/priorformer/runs}
BACKBONE=$RUNS/reference_backbone
VERSATILE=$RUNS/reference_versatile
# stop training 100 min after the data is loaded: leaves time for loading, validation and the save
TRAIN_DATA=$DATA/train
[ "${VARIANT:-staged}" = full ] && TRAIN_DATA="$DATA/train $DATA/train_extra"
TRAIN="python -u -m priorformer.train_v2 --data $TRAIN_DATA --workers $(( SLURM_CPUS_PER_TASK - 1 )) \
       --time-limit 100 --save-every 10"
# generation 2 recipe, every training step: OneCycle (peak 3e-4, 1e-4 when fine-tuning), EMA weights,
# Segment prior noise (±20% global, ±5% per segment, on half the Windows)
RECIPE="--schedule onecycle --ema 0.9999 --seg-jitter 0.2 0.05"

cd "$REPO"
# compute nodes have no git: read the commit from .git
HEAD_REF=$(sed -n 's/^ref: //p' .git/HEAD)
COMMIT=$(cat ".git/$HEAD_REF" 2>/dev/null || grep " $HEAD_REF$" .git/packed-refs | cut -d" " -f1)
echo "=== job ${SLURM_JOB_ID:-local} | commit ${COMMIT:0:7} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case ${VARIANT:-staged} in
    staged)
        EVALUATE="$VERSATILE $BACKBONE"
        if [ ! -f "$BACKBONE/DONE" ]; then
            echo "--- backbone"
            srun --unbuffered $TRAIN $RECIPE --lr 3e-4 --model backbone --out "$BACKBONE" ${TRAIN_ARGS:-}
            exit 0
        elif [ ! -f "$VERSATILE/DONE" ]; then
            echo "--- versatile"
            srun --unbuffered $TRAIN $RECIPE --lr 1e-4 --model versatile --init "$BACKBONE/best.pt" --out "$VERSATILE" ${TRAIN_ARGS:-}
            exit 0
        fi ;;
    scratch)
        EVALUATE=$RUNS/reference_versatile_scratch
        if [ ! -f "$EVALUATE/DONE" ]; then
            echo "--- versatile from scratch"
            srun --unbuffered $TRAIN $RECIPE --lr 3e-4 --model versatile --out "$EVALUATE" ${TRAIN_ARGS:-}
            exit 0
        fi ;;
    full)
        EVALUATE=$RUNS/reference_full_amass
        if [ ! -f "$EVALUATE/DONE" ]; then
            echo "--- versatile from scratch, full AMASS"
            srun --unbuffered $TRAIN $RECIPE --lr 3e-4 --model versatile --epochs ${FULL_EPOCHS:-6} \
                --out "$EVALUATE" ${TRAIN_ARGS:-}
            exit 0
        fi ;;
    *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac

# one camera setting of one model per link
for MODEL in $EVALUATE; do
    for SETTING in centered wild; do
        [ -f "$MODEL/eval_v2/$SETTING.json" ] && continue
        echo "--- evaluate $(basename "$MODEL") $SETTING"
        mkdir -p "$MODEL/eval_v2"
        srun --unbuffered python -u -m priorformer.evaluate_v2 --checkpoint "$MODEL/best.pt" --clips "$DATA/test" \
            --cameras 10 --settings $SETTING --sweep-clips 200 --out "$MODEL/eval_v2/$SETTING.json" ${EVAL_ARGS:-}
        srun --unbuffered python -m priorformer.benchmark --checkpoint "$MODEL/best.pt" --reference \
            > "$MODEL/eval_v2/benchmark_$SETTING.txt"
        exit 0
    done
done
if [ "${VARIANT:-staged}" = full ]; then
    for SETTING in centered wild; do
        [ -f "$EVALUATE/eval_v2/unseen_$SETTING.json" ] && continue
        echo "--- evaluate $(basename "$EVALUATE") $SETTING on the Unseen clips"
        srun --unbuffered python -u -m priorformer.evaluate_v2 --checkpoint "$EVALUATE/best.pt" --clips "$DATA/unseen" \
            --split splits/unseen_clips.tsv --cameras 10 --settings $SETTING --stretch \
            --out "$EVALUATE/eval_v2/unseen_$SETTING.json" ${EVAL_ARGS:-}
        exit 0
    done
fi
echo "--- all done: $EVALUATE"
