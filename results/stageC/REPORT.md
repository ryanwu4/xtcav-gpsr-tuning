# Stage C report — sim-to-sim identifiability closed loop

Date: 2026-08-02. Plan: "STAGE C IMPLEMENTATION PLAN" in the approved plan file
(critic-reviewed; corrections folded in). Status: **all Claude-side work complete;
gates (ii)/(vi)/(vii) pending the Bmad δ-scans** (`make_delta_scans.py`, ~2.5 h,
runbook §7) + `run_stageC_bmadrecovery.py`.

## Gate status (pre-registered in the plan)

| gate | criterion | result | status |
|---|---|---|---|
| (i) self-recovery | pulls/projections recover on the identifiable subspace at λ*, independent-noise targets, both conventions | **rms projection error 0.078** (prior-scaled) over 4 scenarios × 3 seeds on the 5 kept directions; λ*=3e-3; conventions agree (0.44 com vs 0.43 major_lobe pull-err at λ*) | **PASS** |
| (ii) Bmad-truth bias ≤ ~1σ_j on identifiable subspace | replica error bars, C3-relative | **PASS on noise-matched (10k) replicas: surrogate bias ≲0.1σ** vs the C3 baseline (seed-avg rms 0.078 / single-fit 0.121). **Caveat (audit): 100k-target fits show a systematic same-sign 0.5–0.8σ pull on kept dims 3–5 in ALL scenarios incl. truth-zero — uncontrolled (no high-stat C3 control ran).** Stage D's per-step-mean targets are the low-noise case → high-stat control run REQUIRED before quoting a Stage-D bias budget | **PASS (10k-matched); 100k systematic bounded ≤0.8σ, unattributed** |
| (iii) image-gap floor quantified | vs real-vs-sim differences | **TV median 0.77, p90 1.34** (floors: flow 0.056, Bmad-10k 0.09; split err 0.005) — quantified; the like-for-like comparison against real-vs-sim TV needs Stage-D target images (real side), deferred there | **MEASURED** (comparison deferred by necessity) |
| (iv) phase-offset convention | regression vs image cross-check; inert under translation-invariant loss | 24 valtracks: mean +0.025°, σ 0.140°, range [−0.16, +0.61]°; ± internal agreement 0.0015° median (max 0.013°) | **PASS** |
| (v) η_y vs d_nom | formalized | 58.2 mm vs 59.31 mm (−1.9%) | **PASS** |
| (vi) kicker-elimination scenario | ν_y absorbs injected kicker, δ's unbiased within pre-registered tolerance | **PASS baseline-relative (the audit-corrected test): the injection adds ≤0.1σ to any knob pull and ~+0.03 to ν_y vs the no-kicker Bmad scenarios.** The raw max pull 0.72 (S2ER_y, =72 µm) exceeds the pre-registered 2–60 µm scale but appears in EVERY scenario — a generic surrogate-gap artifact, not kicker absorption; the 1.5-pull threshold in finalize_stageC.py was analysis-time, not pre-registered | **PASS (baseline-relative)** |
| (vii) V_eff-error recovery | within replica error bar | **FAIL — and informative**: +2.2 MV injected; 100k fit recovers +0.8 MV with the absorber being **dtr:Q0FF = −6.8% k1** (x-magnification; the compression knobs did NOT move — audit-corrected mechanism); 10k replicas land at 21.4 MV, BELOW the init, one with ν_y=1.31 — image-only V_eff is unstable across noise replicas, not merely biased (weak sensitivity: leverage 0.054, 1.07× above the cut). **Design consequence for Stage D**: V_eff from the externally-measured streak calibration (fixed or tight prior), which the plan's DOF table already prescribed; drift-aliasing caveat carries. | **FAIL → mitigated by design** |

## Pre-registered identifiability cut (BINDING for gate ii — computed before any C4 fit)

Jacobian SVD (central FD under CRN, prior-scaled coordinates, cut = 0.05·s_max):
**5 of 22 directions kept.** Leverage ranking: ν_y 1.00 ≫ L3E 0.42, L2E 0.31,
S2ER_x 0.24, S2EL_x 0.19, … deep null-space: S3ELkG 0.003, Q5FF 0.009, Q4FF 0.022,
L1E 0.031, S1EL_y 0.035, S1ELkG 0.036 (+ dtr:Q0FF). Full record:
`selfrecovery/identifiability.json`. Consequence verified in recovery: individual
knob estimates SHRINK (e.g. S1EL_x truth 1.5σ → 0.0; L2E 1.5σ → 0.0) while the
**projections onto kept directions track truth to 0.01–0.11** — per-knob "bias"
on null dims is shrinkage by design, never a gate quantity (critic finding 2).

## Key numbers (full-statistics run: n=8192/bunch fit, 2-px bandwidth, 200 iters, cosine LR, GPU)

- **λ L-curve**: pull-err-rms 0.66/0.51/0.44/0.45 (com) and 0.70/0.45/0.43/0.46
  (major_lobe) at λ = 3e-4/1e-3/3e-3/1e-2 → **λ* = 3e-3**, flat plateau to 1e-2;
  the two centering conventions give consistent calibration (critic finding 3
  satisfied — Stage D's convention choice won't invalidate λ*).
- **Flipped-target discrete-hypothesis test**: correct pairing 0.071 vs flipped
  1.027 — **14× separation**; select-by-loss is safe.
- **Jitter-blur run** (±0.6° phase ensemble targets, sharp model): max |pull|
  0.17σ — knob-δ bias from Stage D's per-step-mean targets is negligible. **BUT
  ν_y absorbed the blur at −14% (0.862)** — a real Stage-D warning: blurred
  targets contaminate the y-scale nuisance, so Stage D should either include the
  jitter blur in the forward model or treat fitted ν_y as blur-contaminated.
- **Image-gap floor (24 valtracks, δ=0)**: flow-vs-Bmad TV median 0.77 (p90 1.34)
  vs sampling floors 0.056/0.09 — a genuine ~14×-above-noise surrogate residual.
  Charge split, by contrast, agrees to 0.005 median. Interpretation: Stage-D
  δ-attribution must rely on scan-COHERENT image changes (which the fit exploits),
  not absolute image agreement; the C4 bias measurement quantifies how much of
  this residual leaks into δ̂.
- **Optimizer floor**: same-z sanity plateaus at TV 0.081 in 200 iters (the scale
  of the independent-noise floor at n=8192) — Adam does not reach the exact-zero
  same-z solution in a 22-dim nonconvex landscape from a cold start. Direction-
  projected recovery is unaffected; Stage D keeps ensemble restarts.

## Engineering findings (recorded for reuse)

1. **Loss scale**: per-image total-variation distance (unit-sum images), never
   mean-over-all-pixels MAE (O(1e-6), crushed by any sensible prior).
2. **KDE noise floor**: at 1-px bandwidth / n=2048 the independent-z TV floor
   (0.34) swamps 2σ_j signals; **n=16384 + 2-px bandwidth → S/N 2.5** (measured
   scan in `configs/fit_stageC.yaml` comments).
3. **S1EL_y-type dims are ν_y-degenerate on short scans** (found via an initially
   "failing" smoke test that was actually correct MAP behavior); the smoke now
   validates on S2ER_xOffset.
4. **Never use `torch.autograd.functional.jacobian` for tall image Jacobians**
   (one backward per pixel); central FD under CRN is exact and 44 forwards.
5. Device hygiene: `load_dtotr_target` follows the grid's device.
6. **Never give Adam an absolute-volts parameter** (2.2e7 V with lr 0.04 → ~8 V of
   motion in 200 iters; gate (vii) caught it frozen at exactly its init) —
   V_eff is parameterized as a relative factor, like log ν_y.

## Stage-D handoff summary (audit-corrected 2026-08-03)

Stage C verdict: **5 gates pass, 1 measured (comparison deferred), 1 fails
informatively.** Stage D inherits: λ* = 3e-3 (convention-robust); the
identifiable-subspace **procedure** (recompute + re-register the SVD cut on the
real 33-step/8° geometry — the 5 sim directions are local and do not carry
over); a surrogate bias budget of ≲0.1σ noise-matched with an **uncontrolled
≤0.8σ systematic against smooth targets**; per-sample phase offsets
(`phase_offsets.json`); the kicker elimination validated baseline-relative; the
ν_y jitter-sensitivity flag (~15%, one run, sign unexplained — replicate or
model forward, do NOT point-correct); and the σ_j TODO table awaiting
Ryan/ops confirmation.

**Required Stage-D preconditions (from the results audit):** (a) high-statistics-
target C3 control run; (b) ν_y blur replication; (c) SVD re-registration on the
real scan geometry; (d) commit the Stage-C repo state. Audit provenance: every
headline number recomputed from the JSONs and reproduced; identifiability cut
verified bit-exact under current code with timestamps pre-dating all Bmad fits.

## Reproduction

- Self-recovery + identifiability: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src:$TB/src python scripts/run_stageC_selfrecovery.py --iters 200` (~3 h GPU).
- Image gap: `python scripts/run_stageC_imagegap.py` (~10 min GPU).
- Bmad δ-scans (Ryan, tmux): runbook §7; then `python scripts/run_stageC_bmadrecovery.py`.
- Unit tests: 20 in this repo (incl. fit smoke recovering a 2σ_j S2ER_x δ), 24 surrogate-side.
