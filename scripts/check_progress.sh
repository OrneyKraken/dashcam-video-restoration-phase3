#!/bin/bash
# Usage: bash check_progress.sh [run_name]   (default: rvrt_track_a_test)
RUN="${1:-rvrt_track_a_test}"
ROOT="${THESIS_P3_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CSV="$ROOT/results/$RUN/per_clip.csv"
TOTAL=284
DONE=$(($(wc -l < "$CSV") - 1))
START=$(stat "$CSV" | grep Birth | sed 's/ Birth: //')
echo "run: $RUN"
echo "done: $DONE / $TOTAL ($((DONE*100/TOTAL))%)"
python -c "
import datetime
start = datetime.datetime.fromisoformat('$START'.split('.')[0].replace(' ','T'))
now = datetime.datetime.now()
elapsed = (now - start).total_seconds()
done, total = $DONE, $TOTAL
if done > 0:
    pace = elapsed / done
    remaining = (total - done) * pace
    print(f'pace: {pace:.1f}s/clip')
    print(f'remaining: {remaining/3600:.2f} hours')
"
