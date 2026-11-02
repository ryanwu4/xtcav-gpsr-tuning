#!/usr/bin/env bash
# Stage D3 overnight queue (2026-11-01). GPU 1 runs the full-scan fits (started by hand:
# S15673_full S15668_full S15671_full J_full), this script runs the window arms on GPU 0 and
# chains J_win_drift onto GPU 1 once its queue ends. Needs results/stageD3/step_selection.json.
#   bash scripts/stageD3_queue.sh GPU1_PID
set -u
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
LOG=results/stageD3/logs
test -f results/stageD3/step_selection.json || { echo "run stageD3_step_select.py first"; exit 1; }
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src $PY -u scripts/run_stageD3_fit.py --arms J_win --restarts 2 \
    > $LOG/gpu0_jwin.log 2>&1 &&
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src $PY -u scripts/run_stageD3_fit.py --restarts 1 \
    --arms N_full S15673_win S15668_win S15671_win LORO15671 LORO15668 LORO15673 > $LOG/gpu0_win.log 2>&1 &
G0=$!
if [ -n "${1:-}" ]; then
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src $PY -u scripts/run_stageD3_fit.py --arms J_win_drift \
        --restarts 1 > $LOG/gpu1_drift.log 2>&1
fi
wait $G0
