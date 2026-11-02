#!/usr/bin/env bash
# Stage D3 second queue (2026-11-02 03:55): J_win_g moves to GPU 1 after J_win_drift; extra
# restarts (seed 1) of the per-run gain arm and the window-alone arms to size restart scatter.
#   bash scripts/stageD3_queue2.sh GPU1_PID GPU0_PID
set -u
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
LOG=results/stageD3/logs
run() { local gpu=$1 log=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PWD/src $PY -u scripts/run_stageD3_fit.py "$@" > $LOG/$log 2>&1; }
( while kill -0 "$1" 2>/dev/null; do sleep 60; done
  run 1 gpu1_jwin_g.log --arms J_win_g --restarts 1
  run 1 gpu1_jwin_g_s1.log --arms J_win_g --restarts 1 --seed0 1 ) &
( while kill -0 "$2" 2>/dev/null; do sleep 60; done
  run 0 gpu0_win_s1.log --arms S15673_win S15668_win S15671_win --restarts 1 --seed0 1 ) &
wait
