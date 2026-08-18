# Stage D — real-data MAP fit on E300_15673

2026-08-05. First GPSR-style knob-delta fit against real DTOTR2 images: 17 knob-δ
through the xtcav23 flow + δ_tr (Q2/Q1/Q0FF k1) + ν_y + L2_ref nuisance, MAP
(ridge, λ\*=3e-3) on 66 real target images (33 L2-scan steps × both streak signs)
plus the TCAV-off channel. V_eff externally anchored at **27.48 MV** (as-built
segment's numerically-linearized streak cal = measured 14.40; the thin-kick
analytic is −1.2% off). All D0 gates were pre-registered and passed before this
fit ran (`precal/`).

## Headline result

**The fit machinery works end-to-end on real data, and the MAP solution is
physically sane: the fitted deltas are mover-centering–dominated at sub-σ_j
amplitude, exactly the error class this workstream predicted.** The weighted
image residual drops 1.419 → **0.885** (−38%) vs the δ=0 null; the nuisances
alone (ν_y + L2_ref) explain almost nothing (1.404, −1%), so the reduction is
carried by the knob deltas. The remaining residual is **7.1× the split-half
noise floor (0.124)** — dominated by the flow surrogate's image gap (the real
LPS has sharp filamentary structure the flow cannot produce; see gallery), not
by recoverable knob errors.

![gallery](figures/gallery.png)

## Orientation: resolved by the ensemble, not restart 0

The pre-registered orient stage was inconclusive (winner/runner-up TV ratio
1.011 < 2) so BOTH branches were carried through the 8-restart ensemble per
protocol:

| branch | best TV | ν_y | max \|pull\| | restarts converged |
|---|---|---|---|---|
| **flip1_swap1** (rows flipped, signs swapped) | **0.8854** | **0.998** | 0.97 | 6/8 (identical to ~3 decimals) |
| flip0_swap1 | 0.9233 | 0.26–0.45 | 2.66 | 8/8 |

flip1_swap1 wins on loss AND physics: ν_y ≈ 1 (no y-scale correction), δ_tr <
0.1% k1, all pulls < 1σ. The alternative branch can only reach 0.92 by
collapsing the energy axis 3–4× (non-physical) and pulling S1EL_y to +2.5σ.
**flip1_swap1 is adopted as the main hypothesis**; flip0_swap1 is reported
alongside as required.

## Fitted deltas (flip1_swap1, best restart; ensemble range in fig)

| knob | δ̂ (physical) | pull δ̂/σ_j |
|---|---|---|
| S2EL_xOffset | **+95 µm** | +0.95 |
| S2ER_yOffset | −56 µm | −0.56 |
| S1EL_xOffset | −44 µm | −0.44 |
| S1ER_yOffset | +30 µm | +0.30 |
| S1ER_xOffset | +27 µm | +0.27 |
| L3EnergyOffset | −2.3 MeV (−0.04%) | −0.21 |
| L2EnergyOffset | −1.4 MeV (−0.03%) | −0.18 |
| all sextupole/FF-quad strengths | ≤ 0.14σ | ~0 |
| L2_ref shift | −0.17° | (σ=0.3° prior) |
| ν_y | 0.998 → **≈1.06 ± 0.05 after the −6% smooth-target discount** | |
| δ_tr (Q2/Q1/Q0FF) | < 0.1% k1 | |

Sextupole mover offsets at the tens-of-µm scale dominate — consistent with the
prior expectation (mover centering is the known major sim2real error source)
and comfortably inside the σ_j = 100 µm priors. σ_j values are still
TODO-ryan/ops drafts, so pulls are provisional in normalization (not in sign or
ranking).

![pulls](figures/pulls_projections.png)

Identifiable-subspace projections (6 pre-registered directions, D0.3): the
flip1_swap1 projections are small (|median| ≤ 0.93) with tight ensemble spread
on directions 1 and 5; **direction 4 is unconstrained in practice** (spread ~14
across restarts on both branches) and no claim is made on it.

## Pre-registered success criteria (D3.5)

| criterion | result | verdict |
|---|---|---|
| TV reduction vs null > noise floor | 0.534 vs 0.124 | **pass** |
| TV reduction > controlled systematic in TV units | 0.534 vs ≲0.35 (0.42σ max × 0.84 TV/σ max) | **pass** |
| no pull > 3 without a physics story | max 0.97 (main branch) | **pass** |
| ensemble spread < recovered signal on kept projections | directions 0–3,5 yes; direction 4 flagged unconstrained | **pass w/ flag** |
| validation channels not degraded vs null | separation RMS (+ sign) 505→281 µm, (− sign) 390→363 µm; off-channel TV 1.32→0.98 | **pass** |

![tv](figures/per_step_tv.png)

## Held-out-style checks

**Separation curve** (two-lobe x-centroid split, same estimator both sides —
crude; includes non-separable early steps, so not directly comparable to the
Tier-2 8–36 µm band): the fit visibly pulls the + sign curve toward the real
one across the whole scan; the − sign is null-like. The overall 3800→800 µm
compression trend is captured by the model with or without deltas.

![separation](figures/separation_curve.png)

**TCAV-off spot (in-loss residual check, D3.4e):** off-channel TV improves
1.32 → 0.98. The absolute σ comparison in the figure uses full-frame RMS, which
the real side's noise floor inflates (~500 µm vs the Tier-0 Gaussian-core ~150
µm) — the model-vs-real σ levels in fig 5 are therefore not directly
interpretable; only the trend and the TV improvement are.

![offspot](figures/offspot.png)

**Physics deliverable (D3.6, emittance hypothesis):** inconclusive at this
stage — the off-spot core-width comparison needs a Gaussian-core estimator (the
Tier-0 one) rather than full-frame RMS, and the energy-axis over-blur (below)
contaminates the y-width channel. Deferred to the v2 items.

## Sensitivity variants

On the main branch (flip1_swap1, rerun after the ensemble overturned the
orient-stage winner — the pre-registered protocol had run them on flip0_swap1):
see `fit/sensitivity_flip1_swap1.json` (appended when the background run
completes). On flip0_swap1 for the record: Student-t 0.903, V_eff ±10%
0.929/0.926, bandwidth ×0.7/×1.3 0.877/0.987, w_off 0/0.3 0.928/0.922 — the
conclusions above do not hinge on the prior form, V_eff anchor, bandwidth, or
off-channel weight; nothing overturns the branch ranking.

## Caveats and known limitations (v2 backlog)

1. **The 264 µm KDE kernel — not the flow — dominates the visual blur**
   (post-report correction, see `figures/blur_decomposition.png`): at 1-px
   bandwidth the flow renders a sharp S-curve whose core x-FWHM (214 µm) is
   NARROWER than the real mean-target core (274 µm). A Gaussian σ = 264 µm
   kernel alone has FWHM 622 µm — the real image itself refutes it: the
   σ_eff-derived "model-missing smear" (current-profile peak ceiling) conflates
   PSF with geometric/curvature broadening and is an upper bound, not a kernel.
   Empirical kernel implied by the core widths: σ_x ≈ √(274²−214²)/2.355 ≈
   **73 µm** (~3.5× smaller). The true remaining flow gap at 1 px is the
   witness-tail shape and filament detail — the (now much smaller) Stage-E
   fine-tune target.
2. **Isotropic bandwidth over-blurs the energy axis** on top of item 1: PSF_y
   is plausibly ~pixel-scale (30–60 µm), so 264 µm in y is ~5–8× too much. The
   plan's anisotropy-via-per-axis-prescaling (gotcha #1) is not yet
   implemented — with item 1 the top v2 item. It likely also feeds the ν_y
   estimator artifact.
3. **ν_y discount:** fitted ν_y is biased low by −6% ± 5% against smooth
   targets (replicated, `precal/nuy_decision.json`); corrected value ≈ 1.06 ±
   0.05, i.e. the real energy-axis width is consistent with the model within
   the systematic.
4. **Charge-split channel semantics:** targets store the +x-side intensity
   fraction (sign-dependent) while the model reports the drive-charge fraction
   (sign-independent), so one sign row compares against the witness side; the
   term is ≈constant (~0.24) with weak gradient and the w_off/w_q variants show
   the image conclusions are robust to it. v2: image-based same-estimator split.
5. **σ_j table** still carries TODO-ryan/ops on every entry (D0.6).
6. **L1-deltas sensitivity variant not run** — machine_config's
   `L1_delta_per_step_deg` holds step labels, not usable deltas; needs the KLYS
   covariate extraction if wanted.
7. **TSVD-5 companion** realized as reporting-level subspace projections, not
   an in-subspace optimizer.

## Provenance

- Targets: `targets.npz` (66 targets, median 14 shots each after the ±0.2°
  phase cut, split-half floors recorded; 22 camera-frame artifact px repaired at
  source; orientation canonical, hypotheses applied at fit time).
- Segment: `configs/segment_15673.json` (recorded value_match BACTs; entrance
  R12 −20.59, −10.1% vs baseline — critic C1).
- Precal gates: `precal/{register,lambda,control,nuy,nuy_decision}.json`
  (λ\*=3e-3 both anchor conventions; control kept-direction systematic ≤0.42σ;
  6/22 SVD directions kept).
- Fits: `fit/{baselines,orient,ensemble,sensitivity*}.json`; figures + this
  report's numbers: `figures/numbers.json`.
- Fit config: `configs/fit_stageD_15673.yaml`; forward model
  `src/xtcav_gpsr_tuning/{fit,stage_d,targets_real}.py` (chunked-gradient loop:
  the 66-image + off-channel graph exceeds 80 GB in one backward).
