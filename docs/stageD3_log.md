# Stage D3 working log (three-run CNF fit on trusted L2 windows)

Branch `stageD3-multirun` (from `stageD2-cnf` c914b89). Local only, no pushes.
Plan of record: the 2026-11-01 interview (memory `project_stageD3_multirun`).

Question (Ryan, 2026-11-01): fit all three 2026-04-15 E300 L2 scans (TEST_15668, E300_15671, E300_15673), excluding deep-compression steps where the model is known to fail; do different machine setpoints drive sim2real error differently?

## Decisions (2026-11-01 interview)
- Steps: one contiguous L2 window per run from the July sim2real study (+ streak energy-gated separation, real vs aligned FACET2-S2E): window from the scan start to the last passing step before the first sustained (>= 3-step) divergence; a step passes when real sep >= 50 um before full compression and |sim - real| <= max(20 um, 15%).
- Joint fit shares everything (injector incl. one cathode charge, linac, FF, movers, V_eff, nu_y, delta_tr, resolution, one L2 scan gain); per run: L1 phase (readback start, prior sigma 2 deg) and L2 reference (July alignment start, prior sigma 1 deg).
- Surrogate: wide-box CNF (models/c_b3w_ema); arm settings = Stage D2 W1calgr_B (31 knobs, latent mean, convention B, dedicated-cal V_eff, fitted g, fitted resolution), weak ridge lambda_reach 1e-4.
- Charge: shared cathode charge. Injector BPM TMITs (IN10) agree across runs to 0.2%; the 4% spread in TORO LI20 2040 (the Tier-0 charge) first appears at TORO LI11 360 after BC11 and tracks the L1 phase (BC11 chirp-dependent loss).

## Log
- 2026-11-01 23:00 TEST_15668's July sim (L2_ref -38) never reached real steps 0-9 (aligned L2_ref -40.35): reran FACET2-S2E Tier 1 at --l2ref -40.35 (tier1_sim_scans/TEST_15668/l2ref-40.35_value_match, 33 steps, ~40 s/step).
- Real - streak gate fails mid-scan in 15673 (steps 9-18: real - sep 20-140 um vs + 130-200 um; the - images' bands overlap in energy) -> selection uses + streak only (the July primary pairing).
- Targets: convention B (clean row references) for 15671 / 15668 via build_targets_variants.py --dataset (results/stageD3/<run>/targets_rowref_clean.npz). Split-half floor medians 0.075 (15671), 0.115 (15668), 0.119 (15673).
- Code: ForwardModel run_idx (per-run L1/L2 shift vectors; scalar path unchanged). Regression: the new path on 15673 alone reproduces A1calgr_B's best restart exactly (img 0.66701, off 0.39450).
- Cost: 27 s/iteration for 33 steps at 64k particles -> ~54 min per 120-iteration single-run full-scan fit.
- FF quad BACTs agree across the three runs to <0.01%, so one segment (segment_15673.json) serves all three.
