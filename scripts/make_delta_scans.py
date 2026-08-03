"""Stage-C C4: Bmad-truth δ-injected mini-scans (the surrogate-error bias probe).

Four pre-registered scenarios × 7-step L2 mini-scans at 100k macro / 32 threads
(measured ~278 s per full track → ~35 min/scenario, ~2.3 h total). Each step is a
full L0AFEND→DTOTR track (L2 is upstream of the ENDBC20 checkpoint) followed by
±V retracks from that step's checkpoint; the pristine input beam is stashed and
restored per step (the make_validation_tracks lesson).

Scenario 3 injects a KICKER error in truth while the fit has no kicker DOF — the
kicker-elimination gate (ν_y must absorb it). Scenario 4 shifts the retrack
voltage (+10%) with no knob δ — the V_eff-recovery rehearsal (fit with
fit_veff=True; init 22 MV).

Run under bmad-qpad-dev from the xtcav-gpsr-tuning repo root (tmux):

  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
    scripts/make_delta_scans.py [--scenarios mover,energy,kicker,veff] [--nmacro 1e5]

Outputs -> twobunch-s2e-rl/data/xtcav23_delta_scans/<scenario>/
  step{K}_XTCAVF_off.h5, step{K}_DTOTR_{off,plus,minus}.h5, manifest.json
Resume-safe per (scenario, step).
"""

import os

_THREADS = os.environ.get("BMAD_OMP_THREADS", "32")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xtcav_gpsr_tuning import paths  # noqa: E402

paths.ensure_import_roots()
import FACET2_S2E as qs  # noqa: E402
from twobunch_s2e_rl.datagen.sweep_params import XTCAV23_PARAMS  # noqa: E402

CHECKPOINT = "ENDBC20"
N_STEPS = 7
L2_NORM_GRID = np.linspace(0.42, 0.58, N_STEPS)  # matches fit.py base_knobs
FIXED_OVERRIDES = {"Q19851kG": -63.76368742523523, "Q19871kG": 42.49972197426665}
FIT_CFG = Path(__file__).resolve().parents[1] / "configs" / "fit_stageC.yaml"

# δ_true in units of the fit config's σ_j (kicker/veff are absolute)
SCENARIOS = {
    "mover":  {"deltas_sigma": {"S1EL_xOffset": 1.5, "S2ER_xOffset": -2.0,
                                "S2EL_yOffset": 1.0}},
    "energy": {"deltas_sigma": {"L2EnergyOffset": 1.5, "L3EnergyOffset": -1.5}},
    "kicker": {"deltas_abs": {"YC1FFkG": 0.003},  # no kicker DOF in the fit
               "note": "kicker-elimination gate: nu_y must absorb this"},
    "veff":   {"voltage_scale": 1.10,
               "note": "retracks at 24.2 MV; fit with fit_veff=True, init 22 MV"},
}


def mid_knobs() -> dict:
    """All 23 knobs at the normalized box CENTER (0.5) — must match fit.py's base."""
    return {k: (lo + hi) / 2.0 for k, (lo, hi, _) in XTCAV23_PARAMS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="mover,energy,kicker,veff")
    ap.add_argument("--nmacro", type=float, default=1e5)
    ap.add_argument("--voltage", type=float, default=22e6)
    args = ap.parse_args()

    fit_cfg = yaml.safe_load(FIT_CFG.read_text())
    sigma_phys = {k: float(v) for k, v in fit_cfg["sigma_phys"].items()}
    s2e_root = str(paths.FACET2_S2E_ROOT)
    out_root = paths.TWOBUNCH_ROOT / "data" / "xtcav23_delta_scans"
    scratch = out_root / "scratch"
    out_root.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    base_cfg = qs.loadConfig("/setLattice_configs/2024-10-14_twoBunch_baseline.yml",
                             s2e_root)
    tao = qs.initializeTao(
        filePath=s2e_root,
        inputBeamFilePathSuffix=base_cfg["inputBeamFilePathSuffix"],
        csrTF=True, transverseWakes=True,
        numMacroParticles=args.nmacro,
        scratchPath=str(scratch), randomizeFileNames=True,
    )
    pristine = out_root / "input_L0AFEND.h5"
    shutil.copy(tao.activeFilePath, pristine)

    lo_l2, hi_l2, _ = XTCAV23_PARAMS["L2PhaseSet"]
    for scen in [s.strip() for s in args.scenarios.split(",")]:
        spec = SCENARIOS[scen]
        out_dir = out_root / scen
        out_dir.mkdir(exist_ok=True)
        v_amp = args.voltage * spec.get("voltage_scale", 1.0)

        knobs0 = mid_knobs()
        delta_phys = {}
        for k, pulls in spec.get("deltas_sigma", {}).items():
            delta_phys[k] = pulls * sigma_phys[k]
        for k, v in spec.get("deltas_abs", {}).items():
            delta_phys[k] = v
        for k, dv in delta_phys.items():
            knobs0[k] = knobs0[k] + dv

        manifest_path = out_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
            "scenario": scen, "delta_phys": delta_phys, "voltage_V": v_amp,
            "nmacro": int(args.nmacro), "fixed_overrides": FIXED_OVERRIDES,
            "l2_norm_grid": list(map(float, L2_NORM_GRID)), "steps": {},
        }

        for k_step, u in enumerate(L2_NORM_GRID):
            tag = f"step{k_step}"
            files = [out_dir / f"{tag}_DTOTR_{c}.h5" for c in ("off", "plus", "minus")]
            files.append(out_dir / f"{tag}_XTCAVF_off.h5")
            if all(f.exists() for f in files):
                print(f"[{scen}] {tag}: done, skipping", flush=True)
                continue

            knobs = dict(knobs0)
            knobs["L2PhaseSet"] = lo_l2 + u * (hi_l2 - lo_l2)
            shutil.copy(pristine, tao.activeFilePath)
            merged = {**base_cfg, **FIXED_OVERRIDES, **knobs, "trackEnd": "DTOTR"}
            t0 = time.perf_counter()
            qs.setLattice(tao, **merged)
            tao.cmd("set ele XTCAVF VOLTAGE = 0")
            tao.cmd("set ele XTCAVF PHI0 = 0")
            qs.trackBeam(tao, s2e_root, **merged)
            full_s = time.perf_counter() - t0
            P_ckpt = qs.getBeamAtElement(tao, CHECKPOINT)
            qs.makeBeamActiveBeamFile(P_ckpt, tao=tao)

            rec = {"knobs": knobs, "full_track_s": full_s, "cases": {}}
            for case, vsign in (("off", 0.0), ("plus", +1.0), ("minus", -1.0)):
                tao.cmd(f"set ele XTCAVF VOLTAGE = {vsign * v_amp}")
                tao.cmd("set ele XTCAVF PHI0 = 0")
                qs.trackBeam(tao, s2e_root, trackStart=CHECKPOINT,
                             **{**merged, "trackEnd": "DTOTR"})
                P = qs.getBeamAtElement(tao, "DTOTR")
                P.write(str(out_dir / f"{tag}_DTOTR_{case}.h5"))
                rec["cases"][case] = {"n_live": len(P.x)}
                if case == "off":
                    Px = qs.getBeamAtElement(tao, "XTCAVF")
                    Px.write(str(out_dir / f"{tag}_XTCAVF_off.h5"))
            manifest["steps"][tag] = rec
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"[{scen}] {tag}: full {full_s:.0f}s, DTOTR live "
                  f"{rec['cases']['plus']['n_live']}", flush=True)

    print(f"done -> {out_root}")


if __name__ == "__main__":
    main()
