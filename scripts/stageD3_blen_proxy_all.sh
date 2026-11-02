#!/usr/bin/env bash
# Stage D3: BLEN proxies for the July Tier-1 configurations (4 x 33 steps in parallel, 16 threads each).
set -u
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python
LOG=results/stageD3/logs
run() { local log=$1; shift; BMAD_OMP_THREADS=16 PYTHONPATH=/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src \
        MPLBACKEND=Agg $PY -u scripts/stageD3_blen_proxy.py "$@" > $LOG/$log 2>&1; }
run blen_15673.log E300_15673 --l2ref -38 &
run blen_15668.log TEST_15668 --l2ref -40.35 &
run blen_15671_sigma.log E300_15671 --l2ref -38 --l1-abs -22.35 &
run blen_15671_readback.log E300_15671 --l2ref -38 &
wait
