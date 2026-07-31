# xtcav-gpsr-tuning

GPSR-style knob-delta fitting against real FACET-II XTCAV (DTOTR2) images, via a
differentiable XTCAVF→DTOTR Cheetah segment downstream of the two-bunch flow
surrogate. Full design: the approved plan (2026-07-30, critic-reviewed) — see
`results/stageA/REPORT.md` for the Stage-A summary and the lab notebook for the
research context.

## Stage map

| stage | what | status |
|---|---|---|
| A | differentiable XTCAVF→DTOTR transport, validated vs Bmad | **done** (`results/stageA/`) |
| B | 23-dim XTCAVF-entrance surrogate campaign + retrain | **infrastructure done** — sweeps are Ryan's to run: `docs/stageB_runbook.md` |
| C | sim-to-sim identifiability closed loop (CRN, MAP-prior λ calibration) | not started |
| D | real-data fit (per-step target images, MAP δ fit) | not started |
| E | injector LPS DOFs / flow fine-tuning | deferred |

Stage B pieces: `twobunch-s2e-rl` gained the `xtcav23` sweep set (`sweep_params.py`),
config-driven `treaty_points`/`fixed_overrides` in `run_sweep.py`, a `--treaty-point`
arg in `preprocess.py`, and `configs/datagen/xtcav23_{smoke,pilot,full}.yaml`;
`scripts/make_validation_tracks.py` here generates the held-out Bmad DTOTR ground truth.

## Layout

```
src/xtcav_gpsr_tuning/
  paths.py      sibling-repo resolution (Cheetah clone, bridge, twobunch-s2e-rl, FACET2-S2E)
  convert.py    openPMD ParticleGroup <-> cheetah ParticleBeam (t authoritative; READ ITS DOCSTRING)
  segment.py    spec export (PyTao side) + Cheetah Segment builder (crab_cavity->TDC,
                PHI0 rad/2pi passthrough, drift_kick_drift pinned)
scripts/
  export_segment.py    (bmad-qpad-dev) configured Tao -> configs/segment_<optics>.json
                       + fresh Bmad reference beams (off/plus/phi0+-) in data/bmad_reference_<optics>/
  validate_segment.py  (slac-rl) the 5 Stage-A tests -> results/stageA/validation_<optics>.json
configs/    segment specs (committed; deterministic given lattice+config)
tests/      pytest, no Tao needed
data/       (gitignored) reference beams + symlinks to twobunch-s2e-rl/data/xtcav_test_*
results/    stageA validation outputs + REPORT.md
```

## Environments

Never pip-install anything from here; everything goes through PYTHONPATH.

- **Export** (needs pytao + FACET2_S2E): conda `bmad-qpad-dev`
  ```
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/export_segment.py --optics baseline
  ```
- **Validate / fit / tests** (torch stack): conda `slac-rl`
  ```
  PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/validate_segment.py --optics baseline
  /home/rwu4/miniconda3/envs/slac-rl/bin/python -m pytest tests/ -q
  ```
- Cheetah = the 0.8.3-dev source clone at `prev_work/cheetah_analysis/cheetah`
  (put on sys.path by `paths.ensure_import_roots()`; both envs work). The
  upstream `gpsr` repo is a pattern reference only (its `tracking_method="bmadx"`
  is incompatible with Cheetah 0.8.x).

## Conventions that will bite you if you skip them

1. **Saved Bmad beams are exit-plane** (`getBeamAtElement`), and the exported
   spec is start-inclusive (contains the TDC). Never track an already-streaked
   beam through a TDC-containing segment; use `start_after="XTCAVF"` or the
   back-drift bracket (`segment.back_drift`).
2. **tau reference = charge-weighted beam mean**, NOT the lattice `ref_time`
   (`convert.py` docstring has the measurement; the lattice clock is 13.8 ns off
   the bunch and 1 ps = 4.1 deg of X-band RF).
3. **TDC PHI0 is rad/2pi in both Bmad and Cheetah** — passthrough, no degree/sign
   conversion (`set_tdc` takes degrees for Bmad set-command parity).
4. **`drift_kick_drift` everywhere** — Cheetah's default linear maps have no
   per-particle chromaticity and this line is chromatically hair-triggered.
5. Bmad `taylor_map` R16/R36 are reference-orbit-sensitive; validate dispersion
   with the empirical same-estimator regression, not the taylor numbers.
