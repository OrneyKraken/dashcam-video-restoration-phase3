#!/bin/bash
# Usage: bash check_training_progress.sh [run_name]   (default: pretrain_stage1)
RUN="${1:-pretrain_stage1}"
LOG="F:/user4/thesis_p3/logs/${RUN}_console.log"
if [ ! -f "$LOG" ]; then
    echo "no log at $LOG"
    exit 1
fi
tail -1 "$LOG"
