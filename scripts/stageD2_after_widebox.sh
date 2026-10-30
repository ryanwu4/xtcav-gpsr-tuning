#!/bin/bash
# Stage D2, after the wide-box campaign (twobunch-s2e-rl data/widebox_2026-10-30): merged 4096-particle
# table (C_B3's six campaigns + widebox, union frame xtcav23e_l0b2_inj12+xtcav23w_l0b2_inj12w) and its
# full-particle truth; c_b3_ema widened to that frame; C_B3's fine-tune recipe on the merged table;
# export as a fit-only bundle (xtcav-gpsr-tuning/models/c_b3w_ema). The shipped c_b3_ema is untouched.
# New files only. Run from anywhere; GPU for the fine-tune via GPU (default 1).
set -u
TB=/home/rwu4/photoinjector-rl/twobunch-s2e-rl
XG=/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning
CNF=/home/rwu4/photoinjector-rl/facet2-twobunch-cnf
WT=$TB/.worktrees/stageD2_widebox          # union sweep sets (branch stageD2-widebox); data -> ../../data
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
TAG=xtcav23ew_inj_cov_sw_wb_P4096_2026-10-30
RUN=trained/xtcav23ew_C_B3w_2026-10-30
GPU=${GPU:-1}
cd $TB
until grep -q "^Done:" logs/widebox_2026-10-30/run.log; do sleep 120; done
echo "[$(date +%H:%M)] widebox: $(tail -n 1 logs/widebox_2026-10-30/run.log)"
mkdir -p logs/surrogate
PYTHONPATH=$WT/src OMP_NUM_THREADS=1 nice -n 10 $PY -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir xtcav23_full,xtcav23_pilot,injector_full,injector_topup,injector_cov_2026-10-16,compress_sweeps_2026-10-21,widebox_2026-10-30 \
  --sweep-set xtcav23e_l0b2_inj12+xtcav23w_l0b2_inj12w --treaty-point XTCAVF --transmission per_sample --P 4096 \
  --allow-fixed-mismatch num_macro_particles \
  --campaign-json xtcav23_full=results/1_datagen/injector_pilot_2026-10-14/campaign_backfill/xtcav23_full.json \
  --campaign-json xtcav23_pilot=results/1_datagen/injector_pilot_2026-10-14/campaign_backfill/xtcav23_pilot.json \
  --out processed/$TAG.h5 > logs/surrogate/preprocess_$TAG.log 2>&1 || { echo "preprocess failed"; exit 1; }
echo "[$(date +%H:%M)] merged table: $(tail -n 2 logs/surrogate/preprocess_$TAG.log | head -n 1)"
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 $PY scripts/build_truth_table_2026-10-21.py --table processed/$TAG.h5 --workers 48 \
  || { echo "truth failed"; exit 1; }
echo "[$(date +%H:%M)] truth done"
mkdir -p ${RUN}_widened
PYTHONPATH=$XG/src $PY $XG/scripts/stageD2_widen_cnf.py --norm processed/${TAG}_norm.json \
  --out ${RUN}_widened/widened.ckpt || { echo "widen failed"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU $PY -m twobunch_cnf.train --config $CNF/configs/c_b3_ema.yaml \
  --processed processed/$TAG.h5 --truth-h5 processed/${TAG}_truth.h5 \
  --role-map processed/role_map_ftc_s0_2026-10-16.json --init-ckpt ${RUN}_widened/widened.ckpt \
  --out $RUN > logs/surrogate/train_C_B3w_2026-10-30.log 2>&1 || { echo "train failed"; exit 1; }
echo "[$(date +%H:%M)] fine-tune done"
$PY -c "
import json, yaml
k = yaml.safe_load(open('$CNF/src/twobunch_cnf/resources/c_b3_ema/bundle.yaml'))['knobs']
json.dump({d['name']: d['baseline'] for d in k}, open('${RUN}_widened/baseline.json', 'w'), indent=1)"
PYTHONPATH=$WT/src $PY $CNF/scripts/export_bundle.py --ckpt $RUN/checkpoints/final_ema.ckpt \
  --segment-spec configs/segment/segment_baseline.json --baseline-json ${RUN}_widened/baseline.json \
  --name c_b3w_ema --description "Stage D2 fit-only: c_b3_ema fine-tuned on + widebox_2026-10-30" \
  --out $XG/models/c_b3w_ema || { echo "export failed"; exit 1; }
echo "[$(date +%H:%M)] bundle -> $XG/models/c_b3w_ema"
