"""Stage-D2 S2E probe: write a FACET2-S2E campaign (config + hand-built manifest) that
tracks chosen machine states through Impact-T + Bmad at the E300_15673 scan steps, for
rendering through the fit's own screen (xtcav_gpsr_tuning.s2e_probe).

A probe POINT is a machine state: injector deltas (12 Impact-T knobs, physical units,
any size — out-of-box is the point), linac/L0B deltas (physical), and L1 / L2 shifts
[deg] on top of the 15673 conditioning (L1 = −20° anchor, L2 = −37.9° + scan value),
plus an optional "laser_heater" dict (FACET2-S2E addLHmodulation kwargs; needs the
stageD2-widebox tracker, sim/bmad.track(laser_heater=...)). "l2_shift_by_step" {step: deg}
overrides l2_shift_deg per step. "l2_machine" {theta0_deg, phi11_deg} scans L2 the way the
machine's multiknob does (sim/bmad.apply_l2_machine): the lattice sits at the scan centre and
the tracker rotates LI12-14 by this step's offset, LI11 fixed, 14-4/14-6 holding the energy;
optional "hold_delta_deg" (offset from the centre) freezes every quad's field at that L2 state
(the machine LEM'd there) instead of S2E's default re-LEM at every step.
Every point is tracked at every probe step; rows of one step share a linac_id, hence
the Tao random seed, across points (common random numbers for the comparison).

  python scripts/make_s2e_probe.py --name s2e_probe_15673_v1 --points points.json
  then (bmad-qpad-dev, from the stageD2_widebox worktree):
  python -m twobunch_s2e_rl.datagen.injector_campaign <out>/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WT = REPO.parent / "twobunch-s2e-rl" / ".worktrees" / "stageD2_widebox"
sys.path.insert(0, str(WT / "src"))
from twobunch_s2e_rl.datagen.sweep_params import (INJECTOR_KEYS, injector_golden,  # noqa: E402
                                                  resolve_sweep_set)

MC = json.loads(Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results/"
                     "tier1_sim_scans/E300_15673/machine_config.json").read_text())
L1_ANCHOR, L2_REF = -20.0, -37.9
STEPS = [2, 10, 16, 26]          # scan +0.5, +2.5, +4.0, +6.5 deg (0-based target rows)
MID = 16                         # scan centre (+4.0 deg) for machine-like L2 scans


def config(name, out, n_particles, seed, csr_n_bin=None):
    return {
        "name": name, "output_dir": str(out),
        "sweep_set": "xtcav23e_l0b2_inj12", "linac_sweep_set": "xtcav23e_l0b2",
        "injector_sweep_set": "inj12", "n_workers": 32,
        "injector": {"n_particles": int(n_particles), "grid": 64, "arrival": "absolute",
                     "hammersley_burnin": -1},
        "design": {"seed": int(seed), "note": "manifest written by make_s2e_probe.py"},
        "csrTF": True, "transverseWakes": True,
        **({"csr_n_bin": int(csr_n_bin)} if csr_n_bin else {}),
        "baseline_config": "setLattice_configs/2024-10-14_twoBunch_baseline.yml",
        "treaty_points": ["BL11356", "BEGBC20", "XTCAVF", "PENT"],
        "save_beams_at": ["BL11356", "XTCAVF"],
        "fixed_overrides": {"Q19851kG": MC["Q19851kG"], "Q19871kG": MC["Q19871kG"],
                            "trackEnd": "PENT"},
    }


def rows_for(points, steps):
    lkeys, _, _, lbase = resolve_sweep_set("xtcav23e_l0b2")
    golden = injector_golden(INJECTOR_KEYS)
    scan = MC["L2_scan_vals_deg"]
    rows = []
    for si in steps:
        for p in points:
            inj = {k: golden[k] + float(p.get("injector", {}).get(k, 0.0)) for k in INJECTOR_KEYS}
            knobs = {k: float(lbase[k]) + float(p.get("linac", {}).get(k, 0.0)) for k in lkeys}
            knobs["L1PhaseSet"] = L1_ANCHOR + float(p.get("l1_shift_deg", 0.0))
            l2_shift = float(p.get("l2_shift_by_step", {}).get(str(si), p.get("l2_shift_deg", 0.0)))
            knobs["L2PhaseSet"] = L2_REF + l2_shift + float(scan[si])
            extra = {}
            if p.get("l2_machine"):
                # machine-like scan: the lattice sits at the scan centre, the tracker re-phases L2
                knobs["L2PhaseSet"] = L2_REF + l2_shift + float(scan[MID])
                extra["l2_machine"] = {**p["l2_machine"], "delta_deg": float(scan[si] - scan[MID])}
            if p.get("laser_heater"):
                extra["laser_heater"] = p["laser_heater"]
            rows.append({"idx": len(rows), "block": "crossed", "beam": "impact",
                         "injector_point": p["tag"], "injector": inj, "knobs": knobs,
                         "linac_id": int(si), "linac_region": "probe", "step": int(si),
                         "holdout": False, **extra})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--points", required=True, help="json list of probe points")
    ap.add_argument("--n-particles", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--steps", type=int, nargs="*", default=STEPS)
    ap.add_argument("--csr-n-bin", type=int, default=None,
                    help="1D CSR longitudinal bins (campaign csr_n_bin; default: Activate_CSR.tao 40)")
    a = ap.parse_args()
    out = REPO / "data" / a.name
    if (out / "manifest.json").exists():
        sys.exit(f"{out}/manifest.json exists; pick a new --name")
    out.mkdir(parents=True)
    points = json.loads(Path(a.points).read_text())
    (out / "points.json").write_text(json.dumps(points, indent=1))
    (out / "config.yaml").write_text(yaml.safe_dump(config(a.name, out, a.n_particles, a.seed,
                                                           a.csr_n_bin), sort_keys=False))
    rows = rows_for(points, a.steps)
    (out / "manifest.json").write_text(json.dumps(rows, indent=1))
    print(f"{out}: {len(points)} points x {len(a.steps)} steps = {len(rows)} rows")


if __name__ == "__main__":
    main()
