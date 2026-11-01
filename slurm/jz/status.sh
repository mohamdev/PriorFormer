#!/bin/bash
# Status of the research reference model runs on Jean Zay (state, losses, validation and Test MPJPE):
#   bash $WORK/PriorFormer/slurm/jz/status.sh              # once
#   bash $WORK/PriorFormer/slurm/jz/status.sh --watch 60   # refresh every minute, Ctrl+C to quit
exec python3 "$(dirname "$0")/../../scripts/run_status.py" "$@"
