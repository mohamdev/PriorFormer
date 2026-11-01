#!/bin/sh -e
# Research reference model (M0) on the cluster: one stage per submission.
# Run from the clone, with the env active (it is exported to the job):
#   source <env>/bin/activate
#   STAGE=backbone  sh slurm/laas/submit_reference.sh --gpu 1
#   STAGE=versatile sh slurm/laas/submit_reference.sh --gpu 1    # after the backbone finished
#   STAGE=evaluate  sh slurm/laas/submit_reference.sh --gpu 1    # after the versatile model finished
# Cluster paths come from slurm/laas/site.env (see site.env.example); DATA, RESULTS and CHECKPOINT
# override the defaults below. Training stages resume from last.pt: resubmit the same command after
# a timeout. evaluate is an array job, one task per camera setting (centered, wild).

# The helper cds to the script's dir: rerun under an absolute path, since we leave the clone below.
case "$0" in
    /*) ;;
    *) exec sh "$(cd "$(dirname "$0")" && pwd)/$(basename "$0")" "$@" ;;
esac

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
[ -f "$HERE/site.env" ] || { echo "missing $HERE/site.env (copy site.env.example)"; exit 1; }
. "$HERE/site.env"

STAGE=${STAGE:?set STAGE=backbone, versatile or evaluate}
DATA=${DATA:-$DATA_ROOT/amass_shaped}
RUN_ROOT=${RESULTS:-$RESULTS_ROOT}
[ -n "$VIRTUAL_ENV" ] || { echo "activate the env first"; exit 1; }
[ -e "$DATA/train" ] && [ -e "$DATA/test" ] || { echo "no train/ and test/ Clips under $DATA (set DATA)"; exit 1; }

ARRAY="##"
LOGID=%j
case "$STAGE" in
    backbone)
        RESULTS=$RUN_ROOT/reference_backbone
        EXEC="python3 -u -m priorformer.train_v2 --model backbone --data $DATA/train --out $RESULTS --workers 4"
        TIME=2-00:00:00 ;;
    versatile)
        INIT=${INIT:-$RUN_ROOT/reference_backbone/best.pt}
        [ -e "$INIT" ] || { echo "no backbone checkpoint at $INIT (set INIT)"; exit 1; }
        RESULTS=$RUN_ROOT/reference_versatile
        EXEC="python3 -u -m priorformer.train_v2 --model versatile --init $INIT --data $DATA/train --out $RESULTS --workers 4"
        TIME=2-00:00:00 ;;
    evaluate)
        CHECKPOINT=${CHECKPOINT:-$RUN_ROOT/reference_versatile/best.pt}
        [ -e "$CHECKPOINT" ] || { echo "no checkpoint at $CHECKPOINT (set CHECKPOINT)"; exit 1; }
        RESULTS=$(dirname "$CHECKPOINT")/eval_v2
        # task 0: centered cameras, task 1: wild cameras (expanded in the job)
        # a finished setting (its .json exists) is skipped when the array is resubmitted
        EXEC="[ -e $RESULTS/\$SETTING.json ] || python3 -u -m priorformer.evaluate_v2 --checkpoint $CHECKPOINT --clips $DATA/test --cameras 10 --settings \$SETTING --out $RESULTS/\$SETTING.json && python3 -m priorformer.benchmark --checkpoint $CHECKPOINT --reference > $RESULTS/benchmark_\$SETTING.txt"
        ARRAY="#"
        LOGID=%A_%a
        TIME=1-00:00:00 ;;
    *) echo "unknown STAGE=$STAGE"; exit 1 ;;
esac
NAME=pf_ref_$STAGE
mkdir -p "$RESULTS/logs"

# Submit from RESULTS so the helper builds no venv.
cd "$RESULTS"
GPU_REQUIRED=true

. "$SBATCH_HELPER"

jobMessage=$(
sbatch ${sbatchopt} << eof
#!/bin/bash
#SBATCH --job-name="$NAME"
${ARRAY}SBATCH --array=0-1
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --output=$RESULTS/logs/${NAME}_$LOGID.out
#SBATCH --error=$RESULTS/logs/${NAME}_$LOGID.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=50000
#SBATCH --time=$TIME

. "$SBATCH_HELPER"
ulimit -n 65536
${DATASYNC:+$DATASYNC}

SETTING=\$( [ "\${SLURM_ARRAY_TASK_ID:-0}" = 1 ] && echo wild || echo centered )
cd $REPO
echo "python: \$(which python3), commit: \$(git rev-parse --short HEAD), stage: $STAGE"

time sh -c "$EXEC"

eof
)

showSubmitted
