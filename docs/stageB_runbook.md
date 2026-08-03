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

  ```bash
  PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python \
    -m twobunch_s2e_rl.surrogate.diagnostics \
    --ckpt 'trained/xtcav23_pilot/checkpoints/*.ckpt' \
    --processed processed/xtcav23_pilot.h5 \
    --out results/surrogate/xtcav23_pilot/diagnostics
  ```

  The flow must show a clean response on those dims, not noise, before the full
  campaign freezes the conditioning set. Fallback per plan gotcha #13: handoff-plane
  dispersion shims. (Note: `knob_response` now takes its knob list from the table's
  norm frame — the earlier version silently plotted only the first 8 dims; `--samples`
  now auto-picks intact+scraped rows.) Data-level responses were already verified
  strong on all 12 dims in the pilot-gate analysis (16–350× the shot-noise floor).
- **eval.py metrics**: viability AUC / T MAE / spacing-σz R² in the usual place
  (`results/surrogate/` conventions). Two pilot caveats: with only 4/520 nonviable,
  the val-split AUC will often be single-class → `None` (not a failure); and all
  `*_bmag_*` parity rows compare against PENT golden Twiss (β₀=0.5 m) — internally
  consistent true-vs-pred at XTCAVF but physically meaningless labels there; don't
  gate on either.

## 4. Full sweep (10000 + 8, ~2.5–3.5 days at 120 workers)

```bash
cd ~/photoinjector-rl/twobunch-s2e-rl
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  -m twobunch_s2e_rl.datagen.run_sweep configs/datagen/xtcav23_full.yaml
```

## 5. Validation tracks (held-out Bmad DTOTR ground truth for Stage C)

~24 samples × (one ~40 min full track at 100k macro + three retracks) — plan on
~a day in tmux, single process, 32 OpenMP threads (or pass `--nmacro 1e4` for a
fast preview set first):

```bash
cd ~/photoinjector-rl/xtcav-gpsr-tuning
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  scripts/make_validation_tracks.py --campaign xtcav23_full --n 24 --voltage 22e6
```

Notes (from the Stage-B critic review): the script stashes and restores the pristine
L0AFEND input each sample (the checkpoint helper overwrites the active beam file —
without the restore every sample after the first would be garbage). 24 samples is a
deliberate scope cut vs the plan's ~5% — Stage C's flow-vs-Bmad gap estimate carries
24-sample statistics; grow it if that binds. **Stage-C protocol: re-run the Stage-A
kick-vs-z regression per validation sample** (off/plus XTCAVF pairs are not saved here,
but off DTOTR + plus DTOTR support the image-level equivalent) and record the fitted
per-sample TDC phase offset — the beam-mean phase reference varies ±0.6° (1σ) across
the box and shifts ~47 µm under full witness scraping; do not hard-assume the
baseline's −0.001°.

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

- **Handoff plane = XTCAVF element EXIT with V=0** (≡ entrance + an exact drift of the
  TDC length); recorded in norm.json as `treaty_point: XTCAVF`. Fit-time forward model:
  `segment.back_drift(beam, spec["meta"]["tdc_length_m"])` (= 1.0334498 m — always take
  it from the spec, never a rounded literal) then the full TDC-inclusive segment.
  Flow-sampled clouds convert via `convert.cloud_to_beam` (tau = −z exactly; delta from
  total momentum; per-particle charges T_k·Q_k/N) — unit-tested against the t-based path.
- `fixed_overrides`: Q19851/71 at the recorded 2026-04-15 BACTs; `trackEnd: PENT`.
  These are recorded in every sample json for provenance.
- 23-dim box: `sweep_params.XTCAV23_PARAMS` (provenance in its docstring). Movers at
  golden ±0.75 mm — flagged as the one band chosen inside the plan's 0.5–1 mm range;
  adjust there BEFORE the pilot if you disagree.
- XTCAVF specs in the jsons have no BMAG (no golden Twiss at XTCAVF — targetTwiss=None);
  PENT specs are still recorded for cross-checks.

## 7. Stage-C Bmad δ-scans (the one long Stage-C job — tmux, ~2.5 h)

Four pre-registered truth scenarios (mover / energy / kicker-elimination / V_eff)
× 7-step L2 mini-scans at 100k macro, 32 threads (~278 s per full track + 3 short
retracks per step). Resume-safe per (scenario, step):

```bash
cd ~/photoinjector-rl/xtcav-gpsr-tuning
PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
  scripts/make_delta_scans.py
```

Outputs land in `twobunch-s2e-rl/data/xtcav23_delta_scans/<scenario>/` with a
manifest recording the exact injected δ_phys per scenario. The Stage-C fits then
consume them via `scripts/run_stageC_bmadrecovery.py` (slac-rl, GPU).
