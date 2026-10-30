"""Stage-D2: score an S2E probe (make_s2e_probe.py) against the real E300_15673 targets.

For every probe row (point × step): render the saved S2E XTCAVF beam through the fit's
own segment + screen under the calibration of a reference model (default: the δ=0
C_B3 model of the chosen row-correction convention — ν_y 1, δ_tr 0, the anchored V_eff),
and report the per-image TV against the real target of that step (both streak signs,
weighted as in the fit) and the TCAV-off TV. With --cnf, also render the C_B3 CNF at
the same machine state when it lies inside the training box, and report TV(CNF, S2E):
the surrogate's own error at that state.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src python scripts/score_s2e_probe.py \\
      --probe s2e_probe_15673_v1 --conv N --cnf
-> data/<probe>/score_<conv>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_stageD2_reach import TARGET_FILES, get_setup, load_real, make_model  # noqa: E402
from run_stageD_fit import orient_targets  # noqa: E402
from xtcav_gpsr_tuning.s2e_probe import render_s2e  # noqa: E402

CONV_TARGETS = {"D": "deployed", "N": "none", "B": "clean"}


def state_to_model(m, row):
    """Set m's δ (and L1/L2 shifts) to the probe row's machine state at its step; return
    False if any fitted knob would leave the training box (no CNF render then)."""
    keys = m.s.norm["knob_keys"]
    lo, hi = np.array(m.s.norm["knob_low"]), np.array(m.s.norm["knob_high"])
    phys = {**row["knobs"], **row["injector"]}
    with torch.no_grad():
        for i, k in enumerate(m.s.fit_knobs):
            j = keys.index(k)
            u = (phys[k] - lo[j]) / (hi[j] - lo[j])
            d = u - float(m.base[0, j])
            b = float((m.s.bound_hi if d >= 0 else m.s.bound_lo)[i])
            if not (0.0 <= u <= 1.0) or abs(d) >= b:
                return False
            m.delta_raw[i] = float(np.arctanh(d / b)) if b > 0 else 0.0
        # knobs not fitted by the arm must sit at the conditioning baseline
        for k in set(keys) - set(m.s.fit_knobs) - {"L1PhaseSet", "L2PhaseSet"}:
            j = keys.index(k)
            if abs((phys.get(k, np.nan) - lo[j]) / (hi[j] - lo[j]) - float(m.base[0, j])) > 1e-6:
                return False
        m.l1_shift_deg.fill_(row["knobs"]["L1PhaseSet"] + 20.0)
        s = row["step"]
        u2 = (row["knobs"]["L2PhaseSet"] - lo[keys.index("L2PhaseSet")]) / (
            hi[keys.index("L2PhaseSet")] - lo[keys.index("L2PhaseSet")])
        m.l2_shift_deg.fill_((u2 - float(m.base[s, keys.index("L2PhaseSet")])) * m._l2_span_deg)
    return True


@torch.no_grad()
def cnf_images(m, step):
    a, _, c = m.forward_full(rows=[step])
    return a[:, 0].cpu().numpy(), c[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--conv", default="N", choices=list(CONV_TARGETS))
    ap.add_argument("--cnf", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    pdir = REPO / "data" / a.probe
    rows = json.loads((pdir / "manifest.json").read_text())
    real = orient_targets(load_real(dev, CONV_TARGETS[a.conv]), True, True)
    rimg, roff, w = real["images"].cpu().numpy(), real["off"].cpu().numpy(), real["weights"].cpu().numpy()
    ref, _, _ = make_model(f"A1_{a.conv}", dev, seed=0)          # calibration: δ=0 C_B3 model
    out = {"probe": a.probe, "conv": a.conv, "rows": []}
    for r in rows:
        h5 = pdir / f"sample_{r['idx']:05d}_XTCAVF.h5"
        js = json.loads((pdir / f"sample_{r['idx']:05d}.json").read_text()) \
            if (pdir / f"sample_{r['idx']:05d}.json").exists() else {}
        rec = {"idx": r["idx"], "point": r["injector_point"], "step": r["step"],
               "success": bool(js.get("success", False))}
        if not rec["success"] or not h5.exists():
            out["rows"].append(rec)
            continue
        im = render_s2e(ref, h5)
        si = r["step"]
        tv = [float(np.abs(im[k] - rimg[g, si]).sum()) for g, k in enumerate(("plus", "minus"))]
        rec.update(tv_plus=tv[0], tv_minus=tv[1],
                   tv_weighted=float(np.mean([tv[0] * w[0, si], tv[1] * w[1, si]])),
                   tv_off=float(np.abs(im["off"] - roff[si]).sum()),
                   q_drive_nC=im["q_drive_nC"], q_witness_nC=im["q_witness_nC"])
        if a.cnf:
            m, _, _ = make_model(f"A1_{a.conv}", dev, seed=0)
            if state_to_model(m, r):
                ci, co = cnf_images(m, si)
                rec.update(cnf_tv_plus=float(np.abs(ci[0] - rimg[0, si]).sum()),
                           cnf_tv_minus=float(np.abs(ci[1] - rimg[1, si]).sum()),
                           cnf_vs_s2e_plus=float(np.abs(ci[0] - im["plus"]).sum()),
                           cnf_vs_s2e_minus=float(np.abs(ci[1] - im["minus"]).sum()))
            else:
                rec["cnf"] = "out of box"
        np.savez_compressed(pdir / f"render_{a.conv}_{r['idx']:05d}.npz", **{k: v for k, v in im.items()
                                                                           if isinstance(v, np.ndarray)})
        out["rows"].append(rec)
        print(json.dumps(rec), flush=True)
    (pdir / f"score_{a.conv}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
