"""Export the XTCAVF->DTOTR segment spec from a configured Tao + generate fresh
Bmad reference beams (off / plus / phi0+- cases) for Stage-A validation.

Run under the bmad-qpad-dev env from the xtcav-gpsr-tuning repo root:

  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    scripts/export_segment.py --optics baseline [--nmacro 1e4] [--voltage 20e6] \
                              [--phi0-test-deg 2.0] [--skip-tracking]

Outputs:
  configs/segment_<optics>.json                     — element spec + meta (ref energy/times, R-matrices)
  data/bmad_reference_<optics>/beam_{XTCAVF,DTOTR}_{off,plus,phi0p,phi0m}.h5
  data/bmad_reference_<optics>/manifest.json

Mirrors twobunch-s2e-rl/scripts/xtcav_dtotr_test.py: full track L0AFEND->DTOTR with
XTCAV off (establishes the ENDBC20 checkpoint), then 0.5 s retracks per case. The
fresh beams are generated in the SAME Tao session the spec is exported from, so
spec and reference beams are exactly consistent (the July xtcav_test_* artifacts
were tracked with identical settings but a different ISR RNG stream — the
manifest records their results.json medians as a cross-check, not a target).
"""

import os

_THREADS = os.environ.get("BMAD_OMP_THREADS", "32")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtcav_gpsr_tuning import paths
from xtcav_gpsr_tuning.segment import extract_segment_spec, save_spec

paths.ensure_import_roots()

import FACET2_S2E as qs  # noqa: E402  (bmad-qpad-dev only)

CONFIG_FILE = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
CHECKPOINT = "ENDBC20"

# Same set as twobunch-s2e-rl/scripts/xtcav_dtotr_test.py (2024-08-16 GUI values).
CONFIGURATOR_TCAV_OPTICS = {
    "Q5FFkG": -192.1, "Q4FFkG": -162.2, "Q3FFkG": 105.6,
    "Q2FFkG": 120.5, "Q1FFkG": -209.4, "Q0FFkG": 120.5,
    "Q0DkG": -139.3, "Q1DkG": 225.4, "Q2DkG": -139.3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optics", choices=["baseline", "configurator"], default="baseline")
    ap.add_argument("--nmacro", type=float, default=1e4)
    ap.add_argument("--voltage", type=float, default=20e6)
    ap.add_argument("--phi0-test-deg", type=float, default=2.0)
    ap.add_argument("--skip-tracking", action="store_true",
                    help="export the spec only (no beams)")
    args = ap.parse_args()

    s2e_root = str(paths.FACET2_S2E_ROOT)
    out_dir = paths.DATA_DIR / f"bmad_reference_{args.optics}"
    scratch = out_dir / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    paths.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    config = qs.loadConfig(f"/{CONFIG_FILE}", s2e_root)
    if args.optics == "configurator":
        config = {**config, **CONFIGURATOR_TCAV_OPTICS}

    timing = {}
    t0 = time.perf_counter()
    tao = qs.initializeTao(
        filePath=s2e_root,
        inputBeamFilePathSuffix=config["inputBeamFilePathSuffix"],
        csrTF=True,
        numMacroParticles=args.nmacro,
        scratchPath=str(scratch),
        randomizeFileNames=True,
    )
    timing["initializeTao_s"] = time.perf_counter() - t0
    qs.setLattice(tao, **config)  # baseline defaults leave XTCAVvoltage = 0

    # ---- spec export (configured lattice, XTCAV off) --------------------- #
    spec = extract_segment_spec(
        tao, "XTCAVF", "DTOTR",
        meta={"optics": args.optics, "config_file": CONFIG_FILE},
    )
    # Reference R-matrices from Bmad's taylor_map, exit-plane -> exit-plane
    # (getMatrix uses ele.s = exit s; critic C1 plane bookkeeping) plus the
    # entrance-plane variant via startOffset = -L_tdc.
    L_tdc = float(spec["elements"][0]["attrs"]["l"])
    spec["meta"]["tdc_length_m"] = L_tdc
    spec["meta"]["R_exit_to_exit"] = [list(map(float, row)) for row in qs.getMatrix(tao, "XTCAVF", "DTOTR")]
    spec["meta"]["R_entrance_to_exit"] = [
        list(map(float, row))
        for row in qs.getMatrix(tao, "XTCAVF", "DTOTR", startOffset=-L_tdc)
    ]
    spec_path = paths.CONFIGS_DIR / f"segment_{args.optics}.json"
    save_spec(spec, spec_path)
    print(f"spec: {spec_path}  ({len(spec['elements'])} elements, "
          f"E_ref={spec['meta']['ref_energy_eV']/1e9:.4f} GeV, "
          f"ref_time(start)={spec['meta'].get('start_ref_time_s')})")

    if args.skip_tracking:
        return

    # ---- fresh reference beams ------------------------------------------- #
    t0 = time.perf_counter()
    qs.trackBeam(tao, s2e_root, trackEnd="DTOTR", **config)
    timing["trackBeam_full_s"] = time.perf_counter() - t0
    print(f"full track (XTCAV off) in {timing['trackBeam_full_s']:.1f} s")

    P_ckpt = qs.getBeamAtElement(tao, CHECKPOINT)
    qs.makeBeamActiveBeamFile(P_ckpt, tao=tao)

    cases = {
        "off": (0.0, 0.0),
        "plus": (args.voltage, 0.0),
        "phi0p": (args.voltage, +args.phi0_test_deg),
        "phi0m": (args.voltage, -args.phi0_test_deg),
    }
    manifest = {
        "config_file": CONFIG_FILE,
        "optics": args.optics,
        "num_macro_particles": int(args.nmacro),
        "csr": True,
        "transverse_wakes": False,
        "checkpoint": CHECKPOINT,
        "voltage_V": args.voltage,
        "phi0_test_deg": args.phi0_test_deg,
        "spec": str(spec_path),
        "cases": {},
    }
    for case, (V, phi_deg) in cases.items():
        tao.cmd(f"set ele XTCAVF VOLTAGE = {V}")
        tao.cmd(f"set ele XTCAVF PHI0 = {phi_deg / 360.0}")
        t0 = time.perf_counter()
        qs.trackBeam(tao, s2e_root, trackStart=CHECKPOINT, trackEnd="DTOTR", **config)
        timing[f"trackBeam_{case}_s"] = time.perf_counter() - t0
        case_rec = {"voltage_V": V, "phi0_deg": phi_deg}
        for ele in ("XTCAVF", "DTOTR"):
            P = qs.getBeamAtElement(tao, ele)
            P.write(str(out_dir / f"beam_{ele}_{case}.h5"))
            case_rec[ele] = {"n_live": len(P.x)}
        manifest["cases"][case] = case_rec
        print(f"case {case:6s} (V={V/1e6:+.0f} MV, phi0={phi_deg:+.1f} deg): "
              f"{case_rec['DTOTR']['n_live']} live at DTOTR")

    manifest["timing"] = timing
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"reference beams + manifest: {out_dir}")


if __name__ == "__main__":
    main()
