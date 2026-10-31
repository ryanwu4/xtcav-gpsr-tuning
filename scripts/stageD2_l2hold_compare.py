"""Stage-D2: machine-like L2 scan with the quad fields held (no re-LEM) vs re-LEM'd every step.

Probe s2e_l2hold_15673 repeats s2e_l2machine_15673's l2m_A / l2m_B rows (same state, steps and
per-step seeds) with every quad's field frozen at the L2 state the machine was LEM'd in: the scan
centre (_lemC) or the scan start (_lem0). References per row, same step: the same geometry
re-LEM'd every step (S2E's default, s2e_l2machine_15673) and the standard model scan
(s2e_theft_15673). Check: _lem0 at step 0 is the LEM state itself, so it must match its reference.

Per row: image TV (each streak sign and TCAV off; score_s2e_probe.py --conv B renders) against
both references, drive/witness BMAG at BEGBC20 (getBeamSpecs, vs golden Twiss), drive 90% sizes
at XTCAVF and witness transmission to PENT, each next to the re-LEM'd reference.

  python scripts/stageD2_l2hold_compare.py
-> data/s2e_l2hold_15673/compare_B.json (+ a printed table)
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NEW, LEM, REF = (REPO / "data" / p for p in ("s2e_l2hold_15673", "s2e_l2machine_15673",
                                             "s2e_theft_15673"))


def tv(a, b):
    return {k: float(np.abs(a[k] - b[k]).sum()) for k in ("plus", "minus", "off")}


def main():
    idx = lambda d, key: {key(r): r["idx"] for r in json.loads((d / "manifest.json").read_text())}
    lem = idx(LEM, lambda r: (r["injector_point"], r["step"]))
    ref = idx(REF, lambda r: r["step"])
    load = lambda d, i: json.loads((d / f"sample_{i:05d}.json").read_text())
    render = lambda d, i: np.load(d / f"render_B_{i:05d}.npz")
    out = []
    for r in json.loads((NEW / "manifest.json").read_text()):
        p, si = r["injector_point"], r["step"]
        g = p.rsplit("_", 1)[0]
        a, b = load(NEW, r["idx"]), load(LEM, lem[(g, si)])
        rec = {"point": p, "geometry": g, "hold_delta_deg": r["l2_machine"]["hold_delta_deg"],
               "step": si, "vs_relem": tv(render(NEW, r["idx"]), render(LEM, lem[(g, si)])),
               "vs_model_scan": tv(render(NEW, r["idx"]), render(REF, ref[si]))}
        for tag, s in (("held", a["specs"]), ("relem", b["specs"])):
            e20, x = s["BEGBC20"], s["XTCAVF"]
            rec[tag] = {"bmag_drive_x": e20.get("PDrive_BMAG_x"), "bmag_drive_y": e20.get("PDrive_BMAG_y"),
                        "bmag_witness_x": e20.get("PWitness_BMAG_x"),
                        "bmag_witness_y": e20.get("PWitness_BMAG_y"),
                        "sigma90_drive_x_um": 1e6 * x["PDrive_sigmaSI90_x"],
                        "sigma90_drive_y_um": 1e6 * x["PDrive_sigmaSI90_y"],
                        "witness_transmission_PENT": s["PENT"].get("n_live_witness", 0)
                        / max(s["BEGBC20"].get("n_live_witness", 1), 1)}
        out.append(rec)
    (NEW / "compare_B.json").write_text(json.dumps(out, indent=1))
    print(f"{'point':12s} {'step':>4s} | vs re-LEM TV+ TV- off | vs model TV+ TV- | BMAG_x drive "
          f"held/re-LEM | σx90 drive µm held/re-LEM")
    for o in sorted(out, key=lambda o: (o["point"], o["step"])):
        v, m, h, l = o["vs_relem"], o["vs_model_scan"], o["held"], o["relem"]
        print(f"{o['point']:12s} {o['step']:4d} | {v['plus']:6.3f} {v['minus']:6.3f} {v['off']:6.3f} | "
              f"{m['plus']:6.3f} {m['minus']:6.3f} | {h['bmag_drive_x']:6.2f} / {l['bmag_drive_x']:6.2f} | "
              f"{h['sigma90_drive_x_um']:7.1f} / {l['sigma90_drive_x_um']:7.1f}")


if __name__ == "__main__":
    main()
