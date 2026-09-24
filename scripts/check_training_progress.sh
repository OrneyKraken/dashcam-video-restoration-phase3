#!/bin/bash
# Usage: bash check_training_progress.sh [run_name]   (default: pretrain_stage1)
RUN="${1:-pretrain_stage1}"
ROOT="${THESIS_P3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG="$ROOT/logs/${RUN}_console.log"
if [ ! -f "$LOG" ]; then
    echo "no log at $LOG"
    exit 1
fi
tail -1 "$LOG"
