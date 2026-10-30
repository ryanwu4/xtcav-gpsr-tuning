#!/usr/bin/env bash
# Stage-D2 phase-1 reachability queue: one worker per GPU pulling jobs from a shared list
# (results/stageD2/15673/logs/jobs.txt, one job per line: "<name> <script.py> <args>").
# Append lines to the list at any time; a worker exits when the list is empty.
set -uo pipefail
cd /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
LOG=results/stageD2/15673/logs; mkdir -p "$LOG"
JOBS=$LOG/jobs.txt
pop() {  # atomically remove and print the first job line
  exec 9>"$LOG/jobs.lock"; flock 9
  local line; line=$(head -n 1 "$JOBS" 2>/dev/null || true)
  [ -n "$line" ] && sed -i '1d' "$JOBS"
  flock -u 9; echo "$line"
}
worker() {
  local g=$1
  while true; do
    local job; job=$(pop)
    [ -z "$job" ] && break
    local name=${job%% *} args=${job#* }
    echo "$(date +%T) gpu$g start $name" >> "$LOG/progress.txt"
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PWD/src $PY -u $args \
      > "$LOG/$name.log" 2>&1 || echo "$(date +%T) FAILED $name" >> "$LOG/progress.txt"
    echo "$(date +%T) gpu$g done $name" >> "$LOG/progress.txt"
  done
}
worker 0 & worker 1 & wait
echo "$(date +%T) queue empty" >> "$LOG/progress.txt"
