#!/usr/bin/env bash
# run_cal.sh — Run one calibration session with a timestamped output file,
#              then automatically analyze the results.
#
# Usage:
#   scripts/run_cal.sh <run-config> <duration> [--bg]
#
# Arguments:
#   run-config   path to a cal_*.toml or run.toml config file
#   duration     timeout duration: 5m, 1h, 48h, etc. (passed to GNU timeout)
#   --bg         run in the background via nohup (use for runs > ~10 minutes)
#
# Examples:
#   scripts/run_cal.sh config/cal_s1-top_s2-bot.toml 5m
#   scripts/run_cal.sh config/cal_s1-bot_s2-top.toml 5m
#   scripts/run_cal.sh config/cal_s1-top_s2-bot.toml 1h --bg

set -uo pipefail

RUN_CONFIG="${1:?Usage: run_cal.sh <run-config> <duration> [--bg]}"
DURATION="${2:?Usage: run_cal.sh <run-config> <duration> [--bg]}"
BG="${3:-}"

# Derive output names from config key + timestamp
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
    # timeout exits 124 when it kills the process — treat that as normal completion
    timeout "$DURATION" python scripts/log_snr.py --run "$RUN_CONFIG" --out "$CSV" || \
        [[ $? -eq 124 ]] || exit 1
    python scripts/analyze_snr.py --csv "$CSV" --out "$STEM"
}

if [[ "$BG" == "--bg" ]]; then
    nohup bash -c "
        source ~/.bashrc 2>/dev/null || true
        cd $(pwd)
        $(declare -f run_session)
        run_session
    " > "$LOG" 2>&1 &
    echo "PID      : $!"
    echo "Tail log : tail -f $LOG"
else
    run_session
fi
