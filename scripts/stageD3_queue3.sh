#!/usr/bin/env bash
# Stage D3 third queue (2026-11-02 04:55): replaces queue2's GPU-0 branch. After the LORO arms end,
# run the gain-refit LORO variant for all three runs, then the seed-1 window-alone restarts.
#   bash scripts/stageD3_queue3.sh GPU0_PID
set -u
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
LOG=results/stageD3/logs
while kill -0 "$1" 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src $PY -u scripts/stageD3_loro_gain.py 15668 15671 15673 > $LOG/gpu0_lorog.log 2>&1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src $PY -u scripts/run_stageD3_fit.py --arms S15673_win S15668_win S15671_win \
    --restarts 1 --seed0 1 > $LOG/gpu0_win_s1.log 2>&1
