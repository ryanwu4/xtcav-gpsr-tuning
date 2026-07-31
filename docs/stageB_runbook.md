# Stage B runbook — xtcav23 campaign + XTCAVF-entrance surrogate

All sweep/tracking commands run under **bmad-qpad-dev** from the **twobunch-s2e-rl repo
root** (except make_validation_tracks.py, from xtcav-gpsr-tuning); training/preprocess
under **slac-rl**. Everything below is resume-safe (re-run the same command after an
interruption). Expected costs from prior campaigns: ~165 samples/h at 100–120 workers.

## 0. Smoke (already run by Claude, ~10 min)

```bash
cd ~/photoinjector-rl/twobunch-s2e-rl
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  -m twobunch_s2e_rl.datagen.run_sweep configs/datagen/xtcav23_smoke.yaml
```

## 1. Pilot sweep (512 + 8 baseline repeats, ~3–4 h at 100 workers)

```bash
cd ~/photoinjector-rl/twobunch-s2e-rl
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  -m twobunch_s2e_rl.datagen.run_sweep configs/datagen/xtcav23_pilot.yaml
```

## 2. Pilot preprocess + train (slac-rl)

```bash
cd ~/photoinjector-rl/twobunch-s2e-rl
PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python \
  -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir xtcav23_pilot --sweep-set xtcav23 --treaty-point XTCAVF \
  --out processed/xtcav23_pilot.h5

PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python \
  -m twobunch_s2e_rl.surrogate.train \
  --processed processed/xtcav23_pilot.h5 \
  --out trained/xtcav23_pilot --epochs 150
```

## 3. Pilot gates (decide before the full run)

- **Sweep health**: failure rate + witness-viable fraction at XTCAVF from the sample
  jsons (expect high viability — XTCAVF is upstream of the FF/spectrometer scraping;
  the old campaigns' ~31% witness loss happened downstream).
- **Box sanity**: distribution of scraped/failed samples vs the L1/L2 corners (the
  widened longitudinal box intentionally includes early-degradation territory; if the
  failure fraction is large enough to starve density training, revisit the box edges).
- **Knob response on movers/kickers** (the M2-motivated dims):
  `twobunch_s2e_rl.surrogate.diagnostics` knob-response panels on exactly those dims —
  the flow must show a clean response, not noise, before the full campaign freezes the
  conditioning set. Fallback per plan gotcha #13: handoff-plane dispersion shims.
- **eval.py metrics**: viability AUC / T MAE / spacing-σz R² in the usual place
  (`results/surrogate/` conventions).

## 4. Full sweep (10000 + 8, ~2.5–3.5 days at 120 workers)

```bash
cd ~/photoinjector-rl/twobunch-s2e-rl
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  -m twobunch_s2e_rl.datagen.run_sweep configs/datagen/xtcav23_full.yaml
```

## 5. Validation tracks (held-out Bmad DTOTR ground truth for Stage C)

~24 samples × (one ~min full track + three retracks) — a few hours, single process,
32 OpenMP threads:

```bash
cd ~/photoinjector-rl/xtcav-gpsr-tuning
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  scripts/make_validation_tracks.py --campaign xtcav23_full --n 24 --voltage 22e6
```

## 6. Full preprocess + train

```bash
cd ~/photoinjector-rl/twobunch-s2e-rl
PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python \
  -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir xtcav23_full,xtcav23_pilot --sweep-set xtcav23 --treaty-point XTCAVF \
  --out processed/xtcav23_combined.h5

PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python \
  -m twobunch_s2e_rl.surrogate.train \
  --processed processed/xtcav23_combined.h5 \
  --out trained/xtcav23 --epochs 500 --lr-schedule cosine \
  --init-ckpt trained/xtcav23_pilot/checkpoints/<best>.ckpt   # warm start, optional
```

## Conventions baked into the campaign (do not change silently)

- **Handoff plane = XTCAVF element EXIT with V=0** (≡ entrance + exact 1.0334 m drift);
  recorded in norm.json as `treaty_point: XTCAVF`. Fit-time forward model:
  `segment.back_drift(beam, 1.0334)` then the full TDC-inclusive segment.
- `fixed_overrides`: Q19851/71 at the recorded 2026-04-15 BACTs; `trackEnd: PENT`.
  These are recorded in every sample json for provenance.
- 23-dim box: `sweep_params.XTCAV23_PARAMS` (provenance in its docstring). Movers at
  golden ±0.75 mm — flagged as the one band chosen inside the plan's 0.5–1 mm range;
  adjust there BEFORE the pilot if you disagree.
- XTCAVF specs in the jsons have no BMAG (no golden Twiss at XTCAVF — targetTwiss=None);
  PENT specs are still recorded for cross-checks.
