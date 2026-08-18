# Stage D — real-data MAP fit on E300_15673 (v3.1, all gates passing)

2026-08-11. Fit of 17 knob-δ + δ_tr (Q2/Q1/Q0FF k1) + four nuisances (σ_δ, ε_x,
ν_y, L2_ref) to 66 real DTOTR2 images (33 L2-scan steps × both streak signs)
plus the TCAV-off channel, through the xtcav23 flow surrogate and an exact
differentiable Cheetah XTCAVF→DTOTR segment. λ* = 0.01, n = 65536/bunch,
frozen 1-px KDE kernel, V_eff externally anchored at 27.482 MV.

This supersedes v1 (264 µm kernel) and the quarantined v2/v3 runs. Three earlier
attempts were invalidated by defects found in my own code — a quad-k1 state leak,
a missing 1.0334 m handoff back-drift, and a nuisance-prior scaling error. All
are fixed with regression tests; see `../AUDIT_2026-08-07.md` and the archived
`15673_v1_264um/`, `15673_v2_buggy_dtr/`.

## Verdict

**The fitting procedure now works and is well-behaved. It is not yet a
machine-metrology measurement.** It extracts a reproducible, physically sensible,
mover-dominated solution that improves every channel it is scored on, but the
residual model-data gap is ~11× the target noise floor, so the fitted δ̂ are best
read as *the best available correction within this model class*, not as
calibrated per-knob machine errors. The binding limitation is surrogate
fidelity, not the fit.

![assessment](figures/assessment.png)

## Gates — all five pass

| gate | criterion | result |
|---|---|---|
| D0.3 register | identifiable subspace pre-registered | 8 of 24 directions kept |
| D0.4 λ | interior L-curve minimum | λ* = **0.01** (pull-err 0.432/0.400/**0.396**/0.425/0.494 over 1e-3…1e-1) |
| D0.2 control | σ_δ recoverable; bias ≤ ~1σ_j on kept dirs | σ_δ **1.990/1.992** vs truth 2.000 (err_ln −0.002); zero-truth bias max **0.06σ** rms; only Q4FFkG (leverage 0.02) exceeds 0.5σ rms |
| D0.5 ν_y | replicated, unbiased | 0.9996 / 1.0013 / 1.0020 |
| D0 provenance | artifacts match code+config hash | enforced, passing |

The σ_δ gate is the one that failed in v3 and forced this rerun; it now passes to
0.2%, which is what licenses interpreting the fitted σ_δ at all.

## Fit quality

| quantity | value |
|---|---|
| δ=0 null (adopted orientation) | 1.2024 |
| nuisance-only (σ_δ, ε_x, ν_y, L2_ref) | 1.1155 |
| **full MAP** | **0.9582** |
| reduction vs null | **0.244 (20%)** = 2.1× the split-half floor, 20× the controlled bias |
| split-half noise floor | 0.1186 |
| residual / floor, mean over 66 images | **10.9×** (range 2.7–29×, sign-paired floors) |
| images where the fit is worse than the null | **0 / 66** |
| restarts converged | **8/8**, both hypotheses |

Reproducibility is excellent: across all 8 restarts the nuisances agree to 3–4
decimals (σ_δ 1.0881–1.0945, ν_y 1.0184–1.0188) and the projections on the
pre-registered subspace spread by **≤ 0.025**. The optimizer is finding one
well-defined minimum, not a ridge.

## Orientation: resolved

flip1_swap1 wins at **1.44×** over the runner-up, and — the stronger argument —
it is independently favoured **on the δ=0 null alone** (1.202 vs 1.472 / 1.540 /
1.564), where no fitted parameter is involved. The pre-registered 2× ratio is
still not met, so flip0_swap1 was carried through the full ensemble as required;
it lands at 0.958 vs 1.384 and needs σ_δ = 2.14 (vs 1.09) to get there. The
discrete hypothesis is settled in everything but the letter of the rule.

## Fitted parameters

Nuisances, all inside physically sensible ranges:

| | fitted | prior σ | reading |
|---|---|---|---|
| σ_δ | **1.089** | 0.35 (ln) | modest energy-spread deficit; cf. 1.5 implied by the δ=0 y-width diagnostic |
| ε_x | **1.283** | 0.40 (ln) | model x phase space ~28% too small |
| ν_y | **1.019** | 0.03 (ln) | dispersion scale consistent with the −1.9% η_y cross-check |
| L2_ref | **−0.067°** | 0.3° | fitted phase reference essentially confirms −37.9° |
| δ_tr | ≤ **0.51%** k1 | 0.5% | FF quads at their readback accuracy — no absurdity |

Knob deltas — the three leading pulls are all sextupole mover offsets, with
tight ensemble ranges and leverage above the 0.05 cut:

| knob | pull | physical | leverage |
|---|---|---|---|
| S1EL_yOffset | **−1.44σ** | −144 µm | 0.10 |
| S2EL_yOffset | **+1.26σ** | +126 µm | 0.08 |
| S2EL_xOffset | **+1.16σ** | +116 µm | 0.16 |
| S1EL_xOffset | −0.70σ | −70 µm | 0.08 |
| S1ER_yOffset | +0.56σ | +56 µm | 0.06 |
| L3EnergyOffset | −0.34σ | −3.71 MeV (−0.07%) | 0.26 |
| L2EnergyOffset | −0.27σ | −2.20 MeV (−0.05%) | 0.20 |
| sextupole/FF strengths | ≤ 0.23σ | — | mostly < 0.05 (no claim) |

**Mover mis-centering at the 70–145 µm level is the headline result**, which is
the error class this workstream was built to look for. Caveat: σ_j = 100 µm is
still a TODO-ryan/ops draft, so the *pulls* are provisional in normalization
(not in sign or ranking), and pulls > 1σ mean the fit wants more than the assumed
centering accuracy.

## Validation channels (D3.4)

| channel | null | fit | real |
|---|---|---|---|
| separation RMS, + sign (held OUT of the loss) | 506 µm | **215 µm** | — |
| separation RMS, − sign (held out) | 423 µm | **361 µm** | — |
| off-spot σx (in the loss) | 320 µm | 230 µm | **153 µm** |
| off-spot σy (in the loss) | 538 µm | 678 µm | **777 µm** |

Every channel moves toward the data and none degrades. The + sign separation
improves 2.4×, which is a genuine held-out success.

## Where it still falls short

1. **Residual is ~11× the noise floor** (2.7–29× per image). The fit closes 20% of the null's
   discrepancy; 80% remains and is not knob-shaped. This is the flow surrogate's
   image gap plus physics absent from the training campaign (LSC/microbunching
   were off by config), not something more DOFs should be allowed to absorb.
2. **Least gain at deep compression** (+0.20 TV over steps 6–8° vs +0.35 at
   mid-scan), exactly as predicted from the σ_δ step-dependence: at full
   compression the model's y-width is not energy-spread limited, so no global
   σ_δ can close it. The late-step residual is the cleanest single indicator of
   remaining flow error.
3. **ε_x vs off-spot σx tension.** ε_x = 1.283 widens x to help the streaked
   images, but the off-spot σx overshoots the real 153 µm (230 µm fitted, 320 µm
   null). So ε_x is likely absorbing a streaked-plane deficiency at the cost of
   the unstreaked one — a sign the x-plane model error is not a pure emittance
   scale. Worth resolving before ε_x is quoted as an emittance statement.
4. **σ_δ 1.09 vs the 1.5 implied by the δ=0 y-diagnostic.** The fit prefers a
   smaller σ_δ and gets part of the y-width from movers/ν_y instead. Under the
   λ-scaled prior this is data-driven, not shrinkage, but it means the y-width
   attribution between "energy spread" and "mover geometry" is not clean.
5. **Only ~half the knobs are claimable.** 8 of 24 directions kept; 8 knobs sit
   below the 0.05 leverage cut (L1 energy, S1EL/S3EL strengths, Q5/Q4/Q3FF,
   S2ER_y, dtr:Q0FF) and carry no per-knob interpretation.

## Robustness

All pre-registered sensitivity variants leave the picture intact (image loss
0.949–0.969 against the nominal 0.958), and none flips the orientation ranking:

| variant | img | σ_δ | ε_x |
|---|---|---|---|
| nominal | 0.9582 | 1.089 | 1.283 |
| Student-t prior | 0.9485 | 1.071 | 1.264 |
| V_eff ±10% | 0.9617 / 0.9668 | 1.053 / 1.145 | 1.081 / 1.603 |
| bandwidth ×0.7 / ×1.3 | 0.9660 / 0.9553 | 1.092 / 1.092 | 1.323 / 1.236 |
| w_off 0 / 0.3 | 0.9655 / 0.9688 | 1.104 / 1.087 | 1.451 / 1.006 |

ε_x is the least stable parameter (1.01–1.60 across variants), consistent with
item 3 above; σ_δ, ν_y and the leading mover pulls are stable.

## Recommended next steps

1. **σ_j sign-off** (Ryan/ops) — converts the pulls from provisional to
   quantitative. Highest value per unit effort.
2. **Resolve the ε_x / off-spot-σx tension**: try a per-plane (streaked vs
   unstreaked) diagnostic or an x-β mismatch parameterization instead of a pure
   ε scale.
3. **Stage E flow fine-tune** — the only route to the remaining 80%. The
   late-step residual and the step-dependent y-deficit are the targets.
4. Second dataset (E300_15671 or TEST_15668) with δ shared, as the real test of
   whether δ̂ is machine state or dataset-specific overfitting.

## Provenance

Config `configs/fit_stageD_15673.yaml`; targets `targets.npz` (off-shot
amplitude gate + dead-frame floor, off x-rms 115–159 µm across all 33 steps);
segment `configs/segment_15673.json`; precal
`precal/{register,lambda,control,nuy}.json` (hash-stamped); fits
`fit/{orient,baselines,ensemble,sensitivity}.json`; figures + numbers in
`figures/` (`assessment.png`, `diagnostics.png`, `gallery.png`,
`per_step_tv.png`, `pulls_projections.png`, `separation_curve.png`,
`offspot.png`, `kernel_compare.png`, and their `.json` companions).
