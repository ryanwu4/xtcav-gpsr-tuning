#!/usr/bin/env bash
# Stage D2: after the quad-field-hold probe (data/s2e_l2hold_15673) finishes tracking: render its
# rows through the fit's screen, compare with the re-LEM'd machine-like scan and the model scan,
# build one targets file per point, put the gain-recovery fits (A1g_B at lambda*) at the head of
# the GPU queue and start the queue if it is not running. Waits for the campaign's "Done:" line.
set -euo pipefail
XG=/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
P=s2e_l2hold_15673
LOGS=$XG/results/stageD2/15673/logs
cd $XG
until grep -q "^Done:" data/$P/run.log; do sleep 60; done
echo "[$(date +%H:%M)] tracking: $(grep '^Done:' data/$P/run.log)"
CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=$XG/src $PY scripts/score_s2e_probe.py --probe $P --conv B \
  > data/$P/score_B.log 2>&1
echo "[$(date +%H:%M)] rendered"
$PY scripts/stageD2_l2hold_compare.py > data/$P/compare_B.log 2>&1 || echo "compare FAILED (see compare_B.log)"
PTS="l2m_A_lemC l2m_A_lem0 l2m_B_lemC l2m_B_lem0"
for pt in $PTS; do
  PYTHONPATH=$XG/src $PY scripts/build_s2e_targets.py --probe $P --conv B --point $pt
done
T=$XG/results/stageD2/15673/targets_s2e_${P}
new=()
for pt in $PTS; do
  new+=("${pt}_gain scripts/run_stageD2_reach.py --arms A1g_B --restarts 1 --hyp flip1_swap1 --lam 1e-2 --targets-file ${T}_${pt}_B.npz --tag _${pt}_lam1e-2")
done
(
  flock 9
  cp $LOGS/jobs.txt $LOGS/jobs_before_l2hold.txt
  { printf "%s\n" "${new[@]}"; cat $LOGS/jobs.txt; } > $LOGS/jobs.tmp && mv $LOGS/jobs.tmp $LOGS/jobs.txt
) 9>$LOGS/jobs.lock
echo "[$(date +%H:%M)] queued ${#new[@]} fits at the head"
if ! pgrep -f scripts/stageD2_queue.sh > /dev/null; then
  nohup bash scripts/stageD2_queue.sh > /dev/null 2>&1 &
  echo "[$(date +%H:%M)] started the GPU queue"
fi
