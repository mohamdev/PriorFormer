#!/bin/bash
# Login-node loop that keeps one link of slurm/jz/reference_dev.slurm queued until every step is done:
# every few minutes, if no link of the variant (job pf_reference, pf_scratch or pf_full) is queued or running,
# submit the next one. Only one job per loop is ever submitted at a time, so it fits a small
# per-user submit limit. Stops when the evaluation is finished, or after 3 links in a row died in under 10 minutes (read their logs).
# Run it detached from the clone, so it survives logout (one loop per variant: staged, scratch or full):
#   setsid nohup bash slurm/jz/chain_reference.sh <account> [variant] > $SCRATCH/priorformer/runs/logs/chain.log 2>&1 &
# Stop it with: pkill -f "chain_reference.sh <account> [variant]"
set -u
ACCOUNT=${1:?usage: chain_reference.sh <account> [staged|scratch|full]}
VARIANT=${2:-staged}
RUNS=${RUNS:-$SCRATCH/priorformer/runs}
case $VARIANT in
    staged) NAME=pf_reference; FINAL=$RUNS/reference_backbone ;;      # evaluated last
    scratch) NAME=pf_scratch; FINAL=$RUNS/reference_versatile_scratch ;;
    full) NAME=pf_full; FINAL=$RUNS/reference_full_amass ;;
    *) echo "unknown variant $VARIANT"; exit 1 ;;
esac
EVERY=${EVERY:-180}
cd "$(dirname "$0")/../.."
mkdir -p "$RUNS/logs"

fast_fails=0
last=""
while true; do
    DONE_FILE=$FINAL/eval_v2/wild.json
    [ "$VARIANT" = full ] && DONE_FILE=$FINAL/eval_v2/unseen_wild.json
    if [ -f "$DONE_FILE" ]; then
        echo "$(date '+%F %T') all done"
        exit 0
    fi
    if [ -n "$(squeue -u "$USER" -n $NAME -h -o %i 2>/dev/null)" ]; then
        sleep "$EVERY"; continue
    fi
    if [ -n "$last" ]; then
        read -r elapsed state <<< "$(sacct -j "$last" -X -n -P -o ElapsedRaw,State 2>/dev/null | tr '|' ' ')"
        echo "$(date '+%F %T') link $last ended: ${state:-unknown} after ${elapsed:-?} s"
        if [ "${elapsed:-0}" -lt 600 ] && [ "${state%% *}" != COMPLETED ]; then
            fast_fails=$((fast_fails + 1))
        else
            fast_fails=0
        fi
        if [ $fast_fails -ge 3 ]; then
            echo "$(date '+%F %T') 3 links in a row died in under 10 min: stopping, see $RUNS/logs"
            exit 1
        fi
    fi
    job=$(sbatch --parsable --account="$ACCOUNT" --job-name=$NAME --export=ALL,VARIANT=$VARIANT \
          --output="$RUNS/logs/%x_%j.out" --error="$RUNS/logs/%x_%j.out" slurm/jz/reference_dev.slurm 2>&1)
    if [[ "$job" =~ ^[0-9]+$ ]]; then
        echo "$(date '+%F %T') submitted link $job"
        last=$job
    else
        echo "$(date '+%F %T') submit refused, retrying later: $job"
    fi
    sleep "$EVERY"
done
