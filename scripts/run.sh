#!/usr/bin/env bash
# run.sh — Collect data for one run with a timestamped output file,
#           then automatically analyze the results.
#
# Usage:
#   scripts/run.sh <run-config> <duration> [--bg]

set -uo pipefail

RUN_CONFIG="${1:?Usage: run.sh <run-config> <duration> [--bg]}"
DURATION="${2:?Usage: run.sh <run-config> <duration> [--bg]}"
BG="${3:-}"

PY="$HOME/pygpsclient/bin/python"
REPO="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." && pwd )"

KEY=$(basename "$RUN_CONFIG" .toml)
TIMESTAMP=$(date -u +%Y%m%dT%H%M%S)
CSV="data/${KEY}_${TIMESTAMP}.csv"
STEM="data/${KEY}_${TIMESTAMP}"
LOG="data/${KEY}_${TIMESTAMP}.log"

mkdir -p data

echo "Config   : $RUN_CONFIG"
echo "Duration : $DURATION"
echo "CSV      : $CSV"
if [[ "$BG" == "--bg" ]]; then
    echo "Log      : $LOG"
fi

run_session() {
    timeout "$DURATION" "$PY" scripts/log_snr.py --run "$RUN_CONFIG" --out "$CSV" || \
        [[ $? -eq 124 ]] || exit 1
    "$PY" scripts/analyze_snr.py --csv "$CSV" --out "$STEM"
}

if [[ "$BG" == "--bg" ]]; then
    nohup bash -c "
        cd '$REPO'
        timeout '$DURATION' '$PY' scripts/log_snr.py --run '$RUN_CONFIG' --out '$CSV' || \
            [[ \$? -eq 124 ]] || exit 1
        '$PY' scripts/analyze_snr.py --csv '$CSV' --out '$STEM'
    " > "$LOG" 2>&1 &
    echo "PID      : $!"
    echo "Tail log : tail -f $LOG"
else
    run_session
fi
