"""Stage-D2: machine-like vs model L2 scan in true S2E (Impact-T + Bmad), same machine state.

Probe s2e_l2machine_15673 tracks the theft-truth state with the L2 scan done the machine's way
(sim/bmad.apply_l2_machine: LI12-14 rotated, LI11 fixed, 14-4/14-6 holding the energy) for two
LI11 phase geometries (l2m_A in phase, l2m_B on crest), plus each geometry's "_iso" twin: the
model scan at that geometry's vector-sum phase per step (longitudinally identical, so any image
difference between a point and its twin is the L2 energy profile, i.e. transverse). Reference:
the standard model scan of the same state, s2e_theft_15673 (same steps, same per-step seeds).

Per row: image TV against the reference row of the same step (each streak sign and TCAV off;
renders from score_s2e_probe.py --conv B), drive/witness rms σz at XTCAVF and mean energy at
BEGBC20 relative to the reference, and the tracker's L2 solution (vector-sum phase, pair angle,
energy check).

  python scripts/stageD2_l2machine_compare.py
-> data/s2e_l2machine_15673/compare_B.json (+ a printed table)
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NEW, REF = REPO / "data/s2e_l2machine_15673", REPO / "data/s2e_theft_15673"


def tv(a, b):
    return {k: float(np.abs(a[k] - b[k]).sum()) for k in ("plus", "minus", "off")}


def main():
    ref_idx = {r["step"]: r["idx"] for r in json.loads((REF / "manifest.json").read_text())}
    rows = json.loads((NEW / "manifest.json").read_text())
    idx = {(r["injector_point"], r["step"]): r["idx"] for r in rows}
    load = lambda d, i: json.loads((d / f"sample_{i:05d}.json").read_text())
    render = lambda d, i: np.load(d / f"render_B_{i:05d}.npz")
    out = []
    for r in rows:
        si, ri = r["step"], ref_idx[r["step"]]
        a, b = render(NEW, r["idx"]), render(REF, ri)
        js, jr = load(NEW, r["idx"]), load(REF, ri)
        rec = {"point": r["injector_point"], "step": si, **tv(a, b)}
        x, xr = js["specs"]["XTCAVF"], jr["specs"]["XTCAVF"]
        for k in ("sigma_z_rms_drive", "sigma_z_rms_witness"):
            rec[f"d_{k}_um"] = 1e6 * (x[k] - xr[k])
        rec["d_E_BEGBC20_MeV"] = 1e-6 * (js["specs"]["BEGBC20"]["mean_energy_eV"]
                                         - jr["specs"]["BEGBC20"]["mean_energy_eV"])
        if "l2_machine" in js:
            s = js["l2_machine"]
            rec.update(phi_eff_deg=s["phi_eff_deg"], theta_deg=s["theta_deg"],
                       saturated=s["saturated"],
                       energy_check=s["energy_after_eV"] / s["energy_before_eV"] - 1)
        else:
            rec["phi_eff_deg"] = r["knobs"]["L2PhaseSet"]
        out.append(rec)
    # point vs its _iso twin (same step): the energy-profile (transverse) part of the difference
    twins = [{"point": p, "step": si, **tv(render(NEW, i), render(NEW, idx[(p + "_iso", si)]))}
             for (p, si), i in idx.items() if (p + "_iso", si) in idx]
    (NEW / "compare_B.json").write_text(json.dumps({"vs_reference": out, "vs_iso_twin": twins},
                                                   indent=1))
    print(f"{'point':10s} {'step':>4s} {'phi_eff':>8s} {'TV+':>6s} {'TV-':>6s} {'TVoff':>6s} "
          f"{'dσz_d':>7s} {'dσz_w':>7s} {'dE20':>7s}")
    for o in sorted(out, key=lambda o: (o["point"], o["step"])):
        print(f"{o['point']:10s} {o['step']:4d} {o['phi_eff_deg']:8.3f} {o['plus']:6.3f} "
              f"{o['minus']:6.3f} {o['off']:6.3f} {o['d_sigma_z_rms_drive_um']:+7.2f} "
              f"{o['d_sigma_z_rms_witness_um']:+7.2f} {o['d_E_BEGBC20_MeV']:+7.2f}"
              + (f"  θ {o['theta_deg']:.1f} E-check {o['energy_check']:+.1e}"
                 + (" SATURATED" if o["saturated"] else "") if "theta_deg" in o else ""))
    print("point vs iso twin (energy-profile part):")
    for t in sorted(twins, key=lambda t: (t["point"], t["step"])):
        print(f"  {t['point']:8s} step {t['step']:2d}  TV+ {t['plus']:.3f} TV- {t['minus']:.3f} "
              f"off {t['off']:.3f}")


if __name__ == "__main__":
    main()
