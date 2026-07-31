# Stage A report — differentiable XTCAVF→DTOTR segment, validated vs Bmad

Date: 2026-07-30. Plan: `~/.claude/plans/think-about-how-we-dynamic-kettle.md`
(approved, critic-reviewed). All five verification tests ran on **two optics
points** (baseline two-bunch + 2024-08-16 configurator TCAV optics), against
fresh same-session Bmad reference beams (10k macroparticles, CSR on, transverse
wakes off, zero losses; `data/bmad_reference_{baseline,configurator}/`).

## Verdict: PASS — all acceptance criteria met, most by 1–4 orders of margin.

| test | acceptance | baseline | configurator |
|---|---|---|---|
| 1. pure optics (exit→exit), rms/σ per coord | "near-exact" | ≤ 9.6e-5 (x 4.3e-6) | ≤ 1.0e-4 (x 2.6e-5) |
| 2. TDC bracket (+20 MV, PHI0=0), rms/σ | ≲ few % | ≤ 1.3e-3 | ≤ 1.2e-2 |
| 3. PHI0 = ±2° centroids | sign+units pinned | −1.6746/+1.7554 mm vs Bmad −1.6733/+1.7568 | −3.534/+3.484 vs −3.510/+3.508 |
| 4. R12/R11/R22/R34 vs taylor_map | ≲ 1% | ≤ 6e-7 | ≤ 2.7e-6 |
| 4. streak S vs Bmad per-particle fit | ≲ 1% | 5.1e-5 (−11.4292 vs −11.4298) | 3.3e-4 (−22.999 vs −23.006) |
| 4. empirical dispersion dy/dδ | — | 1.1e-7 | 6.9e-7 |
| 5. autograd vs finite differences | ~1e-4 | ≤ 5.8e-11 | ≤ 9.3e-11 |
| 5. batched (vectorized voltage) | runs, consistent | exact (shape [2,10000,7]) | 1.9e-16 |

Unit tests: 10/10 pass (`tests/`, slac-rl env).

## Findings worth remembering

1. **The TDC phase reference is the bunch frame, not the lattice clock.**
   First validation run failed spectacularly (streak sign flipped, −14 mm
   spurious centroid kick) because tau was referenced to `ele_head ref_time` —
   which is 13.8 ns (≈4.13 m) off the bunch mean arrival for the 2024-10 input
   beam. Direct measurement of Bmad's kick (per-particle px difference plus−off
   at XTCAVF exit, regressed vs z) shows slope +0.4788/m = V·k_rf/p0c to 0.02%
   with −0.001° phase offset: **Bmad's zero crossing is the charge-weighted
   beam mean**. `convert.py` now defaults to beam-mean referencing and its
   docstring records the measurement.
2. **Cheetah's single-DKD thick TDC matches Bmad essentially exactly** (streak
   S to 5e-5 baseline / 3.3e-4 configurator). The 1–6% "S_fit vs thin-kick"
   spread in the July artifacts was thin-kick error, not kick-plane ambiguity —
   the thick-cavity effect (−11.43 vs thin −11.34; −23.01 vs thin −24.18) is
   real and Cheetah captures it.
3. **Bmad `taylor_map` R16/R36 are reference-orbit artifacts** in this segment
   (R36 disagrees 5%–53% across optics while the tracked ensembles agree at
   1e-7 via same-estimator dispersion regression and Test 1 y-agreement at
   3e-8 m rms). Never use those two taylor numbers as validation targets.
4. **Configurator bracket residual ~1.1% of beam size** (23 µm centroid at
   R12=−45.9, i.e. <1 DTOTR2 pixel): consistent with the ~µm-level precision
   floor of the beam-mean z-frame assumption amplified by the stronger streak
   optics. This is an absolute-phase nuisance of exactly the kind the Stage-D
   fit's per-dataset timing anchors absorb; shape/slope observables are
   unaffected (S agrees at 3.3e-4).
5. ISR fluctuations are ON in the Bmad references (plan gotcha): visible as
   the small stochastic floor in Test 1 delta/y residuals; irrelevant at
   acceptance thresholds.

## What Stage A delivers for Stages B–D

- `segment.build_segment_from_spec` — batched, differentiable, validated
  forward transport with `set_tdc(V, phase_deg)`; gradients verified w.r.t.
  V and quad k1 (1e-11 vs FD).
- `convert.particlegroup_to_beam` — the coordinate/reference contract, unit-tested.
- `scripts/export_segment.py` — regenerates specs + reference beams for any
  optics config in ~40 s.
- Reference R-matrices (entrance- and exit-plane) and streak calibrations per
  optics in the spec meta + `results/stageA/validation_*.json`.
