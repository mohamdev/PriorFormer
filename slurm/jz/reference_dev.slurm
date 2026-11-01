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
# evaluation (wild cameras). Training stops itself before the walltime and resumes mid-epoch, so the
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
EVAL=$VERSATILE/eval_v2
# stop training 100 min after the data is loaded: leaves time for loading, validation and the save
TRAIN="python -u -m priorformer.train_v2 --data $DATA/train --workers $(( SLURM_CPUS_PER_TASK - 1 )) \
       --time-limit 100 --save-every 10 ${TRAIN_ARGS:-}"

cd "$REPO"
echo "=== job ${SLURM_JOB_ID:-local} | commit $(git rev-parse --short HEAD) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -f "$BACKBONE/DONE" ]; then
    echo "--- backbone"
    srun $TRAIN --model backbone --out "$BACKBONE"
elif [ ! -f "$VERSATILE/DONE" ]; then
    echo "--- versatile"
    srun $TRAIN --model versatile --init "$BACKBONE/best.pt" --out "$VERSATILE"
else
    for SETTING in centered wild; do
        [ -f "$EVAL/$SETTING.json" ] && continue
        echo "--- evaluate $SETTING"
        mkdir -p "$EVAL"
        srun python -u -m priorformer.evaluate_v2 --checkpoint "$VERSATILE/best.pt" --clips "$DATA/test" \
            --cameras 10 --settings $SETTING --out "$EVAL/$SETTING.json" ${EVAL_ARGS:-}
        srun python -m priorformer.benchmark --checkpoint "$VERSATILE/best.pt" --reference > "$EVAL/benchmark_$SETTING.txt"
        exit 0
    done
    echo "--- all done: $EVAL"
fi
