"""Stage-A validation: differentiable XTCAVF->DTOTR Cheetah segment vs Bmad.

Runs the five plan tests (Verification section, corrected per critic review):
  1. Pure optics (exit->exit): beam_XTCAVF_off -> Cheetah (TDC excluded, V irrelevant)
     vs beam_DTOTR_off.
  2. TDC bracket: entrance beam synthesized by back-drifting beam_XTCAVF_off by
     exactly L_tdc, then Cheetah TDC(+V, PHI0=0) -> DTOTR vs beam_DTOTR_plus.
  3. Phase/sign conventions: same bracket at PHI0 = +/- phi0_test_deg vs the
     fresh Bmad phi0p/phi0m beams (pins rad/2pi passthrough + streak sign).
  4. Scalars: numerically-linearized R12/R16/R36 vs the spec's same-session Bmad
     taylor_map (entrance- and exit-plane variants) + streak S vs per-particle fit.
  5. Differentiability + batching: d(sigma_x@DTOTR)/dV and /dk1(Q1FF) vs central
     finite differences; vectorized-voltage batched tracking.

All comparisons state their reference plane; saved Bmad beams are EXIT-plane.

Run (either env; slac-rl proves the cross-env split):
  PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python \
    scripts/validate_segment.py --optics baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from xtcav_gpsr_tuning import paths  # noqa: E402
from xtcav_gpsr_tuning.convert import compare_beams, particlegroup_to_beam  # noqa: E402
from xtcav_gpsr_tuning.segment import (  # noqa: E402
    back_drift,
    build_segment_from_spec,
    element_by_name,
    load_spec,
    numerical_rmatrix,
    set_tdc,
)

paths.ensure_import_roots()
from pmd_beamphysics import ParticleGroup  # noqa: E402

C_LIGHT = 299792458.0


def load_pg(path: Path) -> ParticleGroup:
    return ParticleGroup(str(path))


def sigma_x_weighted(beam) -> torch.Tensor:
    w = beam.particle_charges / beam.particle_charges.sum()
    x = beam.x
    mean = (w * x).sum()
    return ((w * (x - mean) ** 2).sum()).sqrt()


def summarize_compare(cmp: dict, sigma_floor_frac: float = 1e-3) -> dict:
    """Reduce compare_beams output to headline numbers: rms diff / beam std."""
    out = {}
    for coord, d in cmp.items():
        scale = max(abs(d["std_b"]), 1e-30)
        out[coord] = {
            "rms_over_std": d["rms"] / scale,
            "rms": d["rms"],
            "max": d["max"],
            "mean_offset": d["mean_offset"],
            "std_cheetah": d["std_a"],
            "std_bmad": d["std_b"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optics", choices=["baseline", "configurator"], default="baseline")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--ref-dir", default=None, help="fresh Bmad reference beams dir")
    ap.add_argument("--legacy-dir", default=None, help="July xtcav_test_* artifacts (cross-check)")
    args = ap.parse_args()

    spec_path = Path(args.spec) if args.spec else paths.CONFIGS_DIR / f"segment_{args.optics}.json"
    ref_dir = Path(args.ref_dir) if args.ref_dir else paths.DATA_DIR / f"bmad_reference_{args.optics}"
    out_dir = paths.RESULTS_DIR / "stageA"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_spec(spec_path)
    meta = spec["meta"]
    manifest = json.loads((ref_dir / "manifest.json").read_text())
    E_ref = meta["ref_energy_eV"]
    t_ref_start = meta["start_ref_time_s"]  # exit-plane of XTCAVF
    t_ref_end = meta["end_ref_time_s"]      # exit-plane of DTOTR (zero-length monitor)
    L_tdc = meta["tdc_length_m"]
    V = manifest["voltage_V"]
    phi_test = manifest["phi0_test_deg"]

    dtype = torch.float64
    results = {"optics": args.optics, "spec": str(spec_path), "ref_dir": str(ref_dir), "tests": {}}

    seg_exit, rep_exit = build_segment_from_spec(spec, start_after="XTCAVF", dtype=dtype)
    seg_full, rep_full = build_segment_from_spec(spec, dtype=dtype)
    results["segment"] = {
        "exit_plane_elements": rep_exit.n_in,
        "full_elements": rep_full.n_in,
        "cheetah_length_exit_m": rep_exit.cheetah_length,
        "approximated": [list(x) for x in rep_full.approximated],
    }

    def load_pair(case: str):
        Px = load_pg(ref_dir / f"beam_XTCAVF_{case}.h5")
        Pd = load_pg(ref_dir / f"beam_DTOTR_{case}.h5")
        return Px, Pd

    # ------------------------------------------------------------------ #
    # Test 1 — pure optics, exit -> exit, V=0
    # NOTE tau reference: charge-weighted beam mean (ref_time_s=None) — Bmad's
    # effective RF phase reference is the bunch's own z frame (see convert.py
    # contract; the lattice ele_head ref_time is on a different clock).
    # ------------------------------------------------------------------ #
    Px_off, Pd_off = load_pair("off")
    beam_off, _ = particlegroup_to_beam(Px_off, E_ref, None, dtype=dtype)
    out_off = seg_exit.track(beam_off)
    ref_off, _ = particlegroup_to_beam(Pd_off, E_ref, None, dtype=dtype)
    cmp1 = compare_beams(out_off, ref_off, center=("tau",))
    results["tests"]["1_pure_optics_exit_to_exit"] = summarize_compare(cmp1)

    # ------------------------------------------------------------------ #
    # Test 2 — TDC bracket: back-drift entrance beam, TDC(+V, PHI0=0)
    # ------------------------------------------------------------------ #
    entrance = back_drift(beam_off, L_tdc)
    set_tdc(seg_full, V, 0.0)
    out_plus = seg_full.track(entrance)
    _, Pd_plus = load_pair("plus")
    ref_plus, _ = particlegroup_to_beam(Pd_plus, E_ref, None, dtype=dtype)
    cmp2 = compare_beams(out_plus, ref_plus, center=("tau",))
    results["tests"]["2_tdc_bracket_plus"] = summarize_compare(cmp2)
    # Round-trip sanity: V=0 through the full segment == exit-plane result
    set_tdc(seg_full, 0.0, 0.0)
    out_rt = seg_full.track(entrance)
    rt = compare_beams(out_rt, out_off, center=())
    results["tests"]["2b_v0_roundtrip_identity"] = {
        c: {"max": d["max"]} for c, d in rt.items()
    }

    # ------------------------------------------------------------------ #
    # Test 3 — PHI0 conventions at +/- phi_test deg
    # ------------------------------------------------------------------ #
    for case, sign in (("phi0p", +1.0), ("phi0m", -1.0)):
        f = ref_dir / f"beam_DTOTR_{case}.h5"
        if not f.exists():
            continue
        set_tdc(seg_full, V, sign * phi_test)
        out_phi = seg_full.track(entrance)
        ref_phi, _ = particlegroup_to_beam(load_pg(f), E_ref, None, dtype=dtype)
        cmp3 = compare_beams(out_phi, ref_phi, center=("tau",))
        w = out_phi.particle_charges / out_phi.particle_charges.sum()
        results["tests"][f"3_phi0_{case}"] = {
            **summarize_compare(cmp3),
            "x_centroid_cheetah_mm": float((w * out_phi.x).sum() * 1e3),
            "x_centroid_bmad_mm": float((ref_phi.particle_charges / ref_phi.particle_charges.sum() * ref_phi.x).sum() * 1e3),
        }

    # ------------------------------------------------------------------ #
    # Test 4 — scalars: R-matrix + streak S
    # ------------------------------------------------------------------ #
    set_tdc(seg_full, 0.0, 0.0)
    R_num_exit = numerical_rmatrix(seg_exit, E_ref)
    R_num_entr = numerical_rmatrix(seg_full, E_ref)  # V=0: TDC = drift
    R_bmad_exit = np.array(meta["R_exit_to_exit"])
    R_bmad_entr = np.array(meta["R_entrance_to_exit"])

    def r_cmp(Rc, Rb, elems=((0, 1), (0, 5), (2, 5), (0, 0), (1, 1), (2, 3))):
        d = {}
        for i, j in elems:
            key = f"R{i+1}{j+1}"
            d[key] = {
                "cheetah": float(Rc[i, j]),
                "bmad": float(Rb[i, j]),
                "rel_err": float(abs(Rc[i, j] - Rb[i, j]) / max(abs(Rb[i, j]), 1e-12)),
            }
        return d

    results["tests"]["4_rmatrix_exit"] = r_cmp(R_num_exit, R_bmad_exit)
    results["tests"]["4_rmatrix_entrance"] = r_cmp(R_num_entr, R_bmad_entr)
    results["tests"]["4_rmatrix_note"] = (
        "Cheetah R is linearized about the zero orbit; Bmad taylor_map -s is about "
        "the reference orbit. Dispersion-column elements (R16/R36) are "
        "reference-orbit-sensitive — the authoritative map check is Test 1 "
        "(tracked ensemble) and the empirical dispersion below (same estimator "
        "on both sides)."
    )

    # Empirical dispersion at DTOTR: slope of x,y vs delta over the SAME tracked
    # ensemble, Cheetah vs Bmad (incoming correlations identical by construction).
    def disp_slopes(beam):
        p = beam.particles.detach().numpy()
        dlt = p[..., 5]
        return {
            "dx_ddelta": float(np.polyfit(dlt, p[..., 0], 1)[0]),
            "dy_ddelta": float(np.polyfit(dlt, p[..., 2], 1)[0]),
        }

    disp_c = disp_slopes(out_off)
    disp_b = disp_slopes(ref_off)
    results["tests"]["4_empirical_dispersion_off"] = {
        "cheetah": disp_c,
        "bmad": disp_b,
        "rel_err_dy": abs(disp_c["dy_ddelta"] - disp_b["dy_ddelta"])
        / max(abs(disp_b["dy_ddelta"]), 1e-12),
    }

    # Streak S = dx(DTOTR)/dz(XTCAVF-exit) per particle, matching
    # xtcav_dtotr_test.streak_calibration (z = -c*(t - <t>) at XTCAVF exit).
    set_tdc(seg_full, V, 0.0)
    out_S = seg_full.track(entrance)
    z1 = -C_LIGHT * (np.asarray(Px_off.t) - np.average(Px_off.t, weights=Px_off.weight))
    S_cheetah = float(np.polyfit(z1, out_S.x.detach().numpy(), 1)[0])
    z1p = None
    Px_plus = load_pg(ref_dir / "beam_XTCAVF_plus.h5")
    z1p = -C_LIGHT * (np.asarray(Px_plus.t) - np.average(Px_plus.t, weights=Px_plus.weight))
    S_bmad = float(np.polyfit(z1p, np.asarray(Pd_plus.x), 1)[0])
    k_rf = 2 * np.pi * 11.424e9 / C_LIGHT
    S_analytic = float(R_bmad_exit[0, 1] * k_rf * V / E_ref)
    results["tests"]["4_streak_S"] = {
        "cheetah_fit": S_cheetah,
        "bmad_fit": S_bmad,
        "thin_kick_analytic": S_analytic,
        "rel_err_cheetah_vs_bmad": abs(S_cheetah - S_bmad) / abs(S_bmad),
    }

    # ------------------------------------------------------------------ #
    # Test 5 — differentiability + batching
    # ------------------------------------------------------------------ #
    tdc = element_by_name(seg_full, "XTCAVF")
    q1ff = element_by_name(seg_full, "Q1FF")
    set_tdc(seg_full, V, 0.0)
    tdc.voltage.requires_grad_(True)
    q1ff.k1.requires_grad_(True)
    feat = sigma_x_weighted(seg_full.track(entrance))
    feat.backward()
    g_V = float(tdc.voltage.grad)
    g_k1 = float(q1ff.k1.grad)

    def fd(param, h):
        with torch.no_grad():
            orig = param.detach().clone()
            param.copy_(orig + h)
            f_p = float(sigma_x_weighted(seg_full.track(entrance)))
            param.copy_(orig - h)
            f_m = float(sigma_x_weighted(seg_full.track(entrance)))
            param.copy_(orig)
        return (f_p - f_m) / (2 * h)

    fd_V = fd(tdc.voltage, 1e3)      # 1 kV step on 20 MV
    fd_k1 = fd(q1ff.k1, 1e-6)
    results["tests"]["5_gradients"] = {
        "dsigmax_dV": {"autograd": g_V, "fd": fd_V,
                       "rel_err": abs(g_V - fd_V) / max(abs(fd_V), 1e-30)},
        "dsigmax_dk1_Q1FF": {"autograd": g_k1, "fd": fd_k1,
                             "rel_err": abs(g_k1 - fd_k1) / max(abs(fd_k1), 1e-30)},
    }
    tdc.voltage.requires_grad_(False)
    q1ff.k1.requires_grad_(False)

    # Batched: vectorized voltage (0, +V) in one tracking pass
    with torch.no_grad():
        tdc.voltage = torch.tensor([0.0, V], dtype=dtype)
        out_b = seg_full.track(entrance)
        batched_shape = list(out_b.particles.shape)
        x_b = out_b.x
        w = entrance.particle_charges / entrance.particle_charges.sum()
        sig0 = float(((w * (x_b[0] - (w * x_b[0]).sum()) ** 2).sum()).sqrt())
        sig_off = float(sigma_x_weighted(out_off))
        tdc.voltage = torch.tensor(V, dtype=dtype)
    results["tests"]["5_batched"] = {
        "particles_shape": batched_shape,
        "batch0_sigma_x_um": sig0 * 1e6,
        "unbatched_off_sigma_x_um": sig_off * 1e6,
        "rel_err": abs(sig0 - sig_off) / sig_off,
    }

    # Legacy artifact cross-check (different ISR RNG stream — context only)
    if args.legacy_dir or (paths.DATA_DIR / f"xtcav_test_{args.optics}").exists():
        legacy = Path(args.legacy_dir) if args.legacy_dir else paths.DATA_DIR / f"xtcav_test_{args.optics}"
        lg = json.loads((legacy / "results.json").read_text())
        plus = lg["cases"].get("plus", {})
        results["legacy_crosscheck"] = {
            "R12_july": plus.get("R12_XTCAVF_DTOTR_m"),
            "R12_spec_session": float(R_bmad_exit[0, 1]),
            "streak_S_fit_july": plus.get("streak_S_fit"),
            "streak_S_fit_now": S_bmad,
        }

    out_path = out_dir / f"validation_{args.optics}.json"
    out_path.write_text(json.dumps(results, indent=1))

    # ---- console report -------------------------------------------------- #
    print(f"\n=== Stage A validation: {args.optics} ===")
    t1 = results["tests"]["1_pure_optics_exit_to_exit"]
    print("Test 1 (pure optics, exit->exit), rms diff / beam std:")
    for c in ("x", "px", "y", "py", "tau", "delta"):
        print(f"  {c:5s}: {t1[c]['rms_over_std']:.3e}   (rms {t1[c]['rms']:.3e}, "
              f"std_bmad {t1[c]['std_bmad']:.3e})")
    t2 = results["tests"]["2_tdc_bracket_plus"]
    print("Test 2 (TDC bracket, +V PHI0=0), rms diff / beam std:")
    for c in ("x", "px", "y", "py", "tau", "delta"):
        print(f"  {c:5s}: {t2[c]['rms_over_std']:.3e}")
    for k in ("3_phi0_phi0p", "3_phi0_phi0m"):
        if k in results["tests"]:
            t3 = results["tests"][k]
            print(f"Test 3 ({k}): x rms/std {t3['x']['rms_over_std']:.3e}, "
                  f"x centroid cheetah {t3['x_centroid_cheetah_mm']:+.4f} mm "
                  f"vs bmad {t3['x_centroid_bmad_mm']:+.4f} mm")
    print("Test 4 R (entrance->exit):")
    for k, d in results["tests"]["4_rmatrix_entrance"].items():
        print(f"  {k}: cheetah {d['cheetah']:+.6e}  bmad {d['bmad']:+.6e}  rel {d['rel_err']:.2e}")
    s4 = results["tests"]["4_streak_S"]
    print(f"Test 4 streak S: cheetah {s4['cheetah_fit']:.4f}  bmad {s4['bmad_fit']:.4f}  "
          f"rel {s4['rel_err_cheetah_vs_bmad']:.2e}  (thin-kick {s4['thin_kick_analytic']:.4f})")
    d4 = results["tests"]["4_empirical_dispersion_off"]
    print(f"Test 4 empirical dy/ddelta: cheetah {d4['cheetah']['dy_ddelta']:+.5f}  "
          f"bmad {d4['bmad']['dy_ddelta']:+.5f}  rel {d4['rel_err_dy']:.2e}")
    g5 = results["tests"]["5_gradients"]
    print(f"Test 5 grads: dV rel {g5['dsigmax_dV']['rel_err']:.2e}, "
          f"dk1 rel {g5['dsigmax_dk1_Q1FF']['rel_err']:.2e}; "
          f"batched shape {results['tests']['5_batched']['particles_shape']}, "
          f"batch-off vs unbatched rel {results['tests']['5_batched']['rel_err']:.2e}")
    print(f"\nfull results: {out_path}")


if __name__ == "__main__":
    main()
