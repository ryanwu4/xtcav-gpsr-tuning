"""Stage D3: simulated bunch-length-monitor (BLEN) proxies along the three E300 L2 scans.

The real DAQ records three CSR/pyrometer bunch-length monitors (BLEN, raw counts, higher =
shorter): after BC11 (LI11 356/359), after BC14 (LI14 888) and in the final focus (LI20 3014).
Their raw counts are uncalibrated, so the comparison uses shape only: where along the L2 scan
each signal peaks and how it rises. This script re-tracks the FACET2-S2E Tier-1 configurations
(same machine_config.json, same L1/L2/quad overrides as the July sims) and records, at the
markers next to each monitor (ENDBC11_2, ENDBC14_2, MFFF), the bunch length and a
parameter-free coherent-power proxy, int lambda(z)^2 dz (proportional to 1/sigma_z for a
Gaussian), plus high-pass versions of it (form factor above 0.3 and 1 THz).

Run (bmad-qpad-dev env; from xtcav-gpsr-tuning root):
  BMAD_OMP_THREADS=16 PYTHONPATH=/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/stageD3_blen_proxy.py E300_15673 \
    --l2ref -38 [--l1-abs -22.35] [--tag NAME] [--steps 1,17,33] [--check]
-> results/stageD3/blen_proxy/<dataset>_<tag>.json
"""

import os

_THREADS = os.environ.get("BMAD_OMP_THREADS", "16")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import FACET2_S2E as qs  # noqa: E402

TB = Path("/home/rwu4/photoinjector-rl/twobunch-s2e-rl/scripts")
sys.path.insert(0, str(TB))
from xtcav_dtotr_test import CONFIG_FILE, S2E_ROOT, set_xtcav  # noqa: E402
from xtcav_sim_scan import OUT_ROOT, step_overrides  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "stageD3" / "blen_proxy"
MARKERS = ("ENDBC11_2", "ENDBC14_2", "MFFF")   # next to BLEN LI11 356, LI14 888, LI20 3014
C = 299792458.0


def proxies(P):
    """Bunch-length numbers of one beam: rms and 90%-core sigma_z, and int lambda^2 dz (unbiased
    pair estimator on a fine histogram) with high-pass variants from the form factor."""
    z = np.asarray(P.z, float)
    w = np.asarray(P.weight, float)
    z = z - np.average(z, weights=w)
    sig = float(np.sqrt(np.average(z ** 2, weights=w)))
    dz = sig / 200.0
    n_bins = int(np.ceil((z.max() - z.min()) / dz)) + 8
    edges = z.min() - 4 * dz + dz * np.arange(n_bins + 1)
    h, _ = np.histogram(z, edges, weights=w)
    W = w.sum()
    lam2 = float((np.sum(h ** 2) - np.sum(w ** 2)) / (W ** 2 * dz))   # [1/m]
    # form factor |F(k)|^2 of the normalized profile; int lambda^2 dz = (1/pi) int_0^inf |F|^2 dk
    pad = 8 * len(h)
    F = np.fft.rfft(h / W, n=pad)
    k = 2 * np.pi * np.fft.rfftfreq(pad, d=dz)
    F2 = np.abs(F) ** 2 - np.sum(w ** 2) / W ** 2                       # shot-noise floor removed
    dk = k[1] - k[0]
    out = {"sigma_z_rms_um": 1e6 * sig, "sigma_z_core_um": 1e6 * qs.smallestIntervalImpliedSigma(z),
           "int_lambda2_per_mm": 1e-3 * lam2, "n_live": int(len(z))}
    for f_thz in (0.3, 1.0):
        kc = 2 * np.pi * f_thz * 1e12 / C
        out[f"hp{f_thz:g}THz_per_mm"] = float(1e-3 * F2[k >= kc].sum() * dk / np.pi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--l2ref", type=float, required=True)
    ap.add_argument("--l1-abs", type=float, default=None)
    ap.add_argument("--quad-map", default="value_match")
    ap.add_argument("--nmacro", type=float, default=1e4)
    ap.add_argument("--steps", default="all")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--l1-sweep", default=None,
                    help="LO:HI:STEP absolute L1 phases; tracks to ENDBC11_2 only (BC11 depends on L1 alone)")
    ap.add_argument("--check", action="store_true",
                    help="step 1 only: compare the ENDBC11_2 beam from a short track with the one "
                         "left after the long track (tests that upstream markers survive)")
    args = ap.parse_args()

    mc = json.load(open(OUT_ROOT / args.dataset / "machine_config.json"))
    tag = args.tag or (f"l2ref{args.l2ref:+.2f}" + (f"_l1abs{args.l1_abs:+.2f}" if args.l1_abs is not None else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / "scratch" / f"{args.dataset}_{tag}"
    scratch.mkdir(parents=True, exist_ok=True)
    n_steps = len(mc["L2_scan_vals_deg"])
    steps = ([1] if args.check else list(range(1, n_steps + 1)) if args.steps == "all"
             else [int(s) for s in args.steps.split(",")])

    base = qs.loadConfig(f"/{CONFIG_FILE}", str(S2E_ROOT))
    tao = qs.initializeTao(filePath=str(S2E_ROOT), inputBeamFilePathSuffix=base["inputBeamFilePathSuffix"],
                           csrTF=True, numMacroParticles=args.nmacro, scratchPath=str(scratch),
                           randomizeFileNames=True)
    pristine = scratch / "input_L0AFEND.h5"
    shutil.copy(tao.activeFilePath, pristine)

    res = {"dataset": args.dataset, "tag": tag, "l2ref": args.l2ref, "l1_abs": args.l1_abs,
           "quad_map": args.quad_map, "nmacro": int(args.nmacro), "markers": MARKERS, "steps": {}}
    path = OUT / f"{args.dataset}_{tag}{'_check' if args.check else ''}.json"
    t_run = time.perf_counter()
    if args.l1_sweep:
        lo, hi, st = (float(x) for x in args.l1_sweep.split(":"))
        res["l1_sweep"] = []
        for l1 in np.arange(lo, hi + 1e-9, st):
            shutil.copy(pristine, tao.activeFilePath)
            cfg = step_overrides(mc, base, 1, args.l2ref, args.quad_map, l1_abs=float(l1))
            qs.setLattice(tao, **cfg)
            qs.trackBeam(tao, str(S2E_ROOT), trackEnd="ENDBC11_2", **cfg)
            row = {"L1PhaseSet": float(l1), "ENDBC11_2": proxies(qs.getBeamAtElement(tao, "ENDBC11_2"))}
            res["l1_sweep"].append(row)
            print(f"L1 {l1:+.2f}: sz {row['ENDBC11_2']['sigma_z_rms_um']:.1f} um "
                  f"L2 {row['ENDBC11_2']['int_lambda2_per_mm']:.3g}/mm", flush=True)
        path = OUT / f"l1sweep_BC11.json"
        path.write_text(json.dumps(res, indent=1))
        return
    for step in steps:
        shutil.copy(pristine, tao.activeFilePath)
        cfg = step_overrides(mc, base, step, args.l2ref, args.quad_map, l1_abs=args.l1_abs)
        qs.setLattice(tao, **cfg)
        set_xtcav(tao, 0.0, 0.0)
        t0 = time.perf_counter()
        if args.check:
            qs.trackBeam(tao, str(S2E_ROOT), trackEnd="ENDBC11_2", **cfg)
            res["check_short_ENDBC11_2"] = proxies(qs.getBeamAtElement(tao, "ENDBC11_2"))
            shutil.copy(pristine, tao.activeFilePath)
        qs.trackBeam(tao, str(S2E_ROOT), trackEnd="MFFF", **cfg)
        row = {"L1PhaseSet": cfg["L1PhaseSet"], "L2PhaseSet": cfg["L2PhaseSet"],
               "scan_deg": mc["L2_scan_vals_deg"][step - 1], "t_track_s": time.perf_counter() - t0}
        for mk in MARKERS:
            row[mk] = proxies(qs.getBeamAtElement(tao, mk))
        res["steps"][str(step)] = row
        print(f"step {step}: {row['t_track_s']:.0f} s  " + "  ".join(
            f"{mk} sz {row[mk]['sigma_z_rms_um']:.1f} um L2 {row[mk]['int_lambda2_per_mm']:.3g}/mm"
            for mk in MARKERS), flush=True)
        path.write_text(json.dumps(res, indent=1))
    res["total_s"] = time.perf_counter() - t_run
    if args.check:
        print("short-track ENDBC11_2:", res["check_short_ENDBC11_2"])
        print("long-track  ENDBC11_2:", res["steps"]["1"]["ENDBC11_2"])
    path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
