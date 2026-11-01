#!/bin/bash
# Login-node loop that keeps one link of slurm/jz/reference_dev.slurm queued until every step is done:
# every few minutes, if no pf_reference job is queued or running, submit the next link. Only one
# job is ever submitted at a time, so it fits a small per-user submit limit. Stops when the
# evaluation is finished, or after 3 links in a row died in under 10 minutes (read their logs).
# Run it detached from the clone, so it survives logout:
#   setsid nohup bash slurm/jz/chain_reference.sh <account> > $SCRATCH/priorformer/runs/logs/chain.log 2>&1 &
# Stop it with: pkill -f chain_reference.sh
set -u
ACCOUNT=${1:?usage: chain_reference.sh <account>}
RUNS=${RUNS:-$SCRATCH/priorformer/runs}
NAME=pf_reference
EVERY=${EVERY:-180}
cd "$(dirname "$0")/../.."
mkdir -p "$RUNS/logs"

fast_fails=0
last=""
while true; do
    if [ -f "$RUNS/reference_versatile/eval_v2/wild.json" ]; then
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
    job=$(sbatch --parsable --account="$ACCOUNT" --output="$RUNS/logs/%x_%j.out" slurm/jz/reference_dev.slurm 2>&1)
    if [[ "$job" =~ ^[0-9]+$ ]]; then
        echo "$(date '+%F %T') submitted link $job"
        last=$job
    else
        echo "$(date '+%F %T') submit refused, retrying later: $job"
    fi
    sleep "$EVERY"
done
