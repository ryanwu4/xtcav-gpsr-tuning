#!/usr/bin/env bash
# Stage D2: after the machine-like L2 scan probe (data/s2e_l2machine_15673) finishes tracking:
# render its rows through the fit's screen, compare with the standard-scan theft probe, build one
# targets file per point, and put the gain-recovery fits (A1g_B at lambda* on each point, A1_B on
# l2m_A) at the head of the GPU queue. Run from anywhere; waits for the campaign's "Done:" line.
set -euo pipefail
XG=/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
P=s2e_l2machine_15673
LOGS=$XG/results/stageD2/15673/logs
cd $XG
until grep -q "^Done:" data/$P/run.log; do sleep 60; done
echo "[$(date +%H:%M)] tracking: $(grep '^Done:' data/$P/run.log)"
CUDA_VISIBLE_DEVICES=${GPU:-1} PYTHONPATH=$XG/src $PY scripts/score_s2e_probe.py --probe $P --conv B \
  > data/$P/score_B.log 2>&1
echo "[$(date +%H:%M)] rendered"
$PY scripts/stageD2_l2machine_compare.py
for pt in l2m_A l2m_A_iso l2m_B l2m_B_iso; do
  PYTHONPATH=$XG/src $PY scripts/build_s2e_targets.py --probe $P --conv B --point $pt
done
T=$XG/results/stageD2/15673/targets_s2e_${P}
new=()
for pt in l2m_A l2m_B l2m_A_iso l2m_B_iso; do
  new+=("${pt}_gain scripts/run_stageD2_reach.py --arms A1g_B --restarts 1 --hyp flip1_swap1 --lam 1e-2 --targets-file ${T}_${pt}_B.npz --tag _${pt}_lam1e-2")
done
new+=("l2m_A_nogain scripts/run_stageD2_reach.py --arms A1_B --restarts 1 --hyp flip1_swap1 --lam 1e-2 --targets-file ${T}_l2m_A_B.npz --tag _l2m_A_lam1e-2")
(
  flock 9
  cp $LOGS/jobs.txt $LOGS/jobs_before_l2machine.txt
  { printf "%s\n" "${new[@]}"; cat $LOGS/jobs.txt; } > $LOGS/jobs.tmp && mv $LOGS/jobs.tmp $LOGS/jobs.txt
) 9>$LOGS/jobs.lock
echo "[$(date +%H:%M)] queued ${#new[@]} fits at the head"
