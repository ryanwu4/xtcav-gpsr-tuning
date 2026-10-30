"""Stage-D2 metrology report: restart agreement and split consistency of a λ* fit, in the
registered identifiable subspace (Stage D's protocol, with the projection Stage D left open).

Inputs (results/stageD2/15673/reach/): restarts <arm><tag>_r<i>.json, split fits
<arm><tag>_fit_{even,odd,low,deep}.json; the registration register_<arm><tag>_r0.json
(stageD2_jacobian_alignment.py at the r0 state); the S2E theft control at the same λ for
recovery factors (A1_B_theft_lam<λ>.json, truth data/s2e_theft_15673/points.json).

Coordinates: every parameter in its D2 prior σ (knobs δ/σ_j, L1/L2 shifts / their σ, the
L2 scan gain (g − 1) / 0.25 when fitted), the
frame the registration's SVD was taken in. Pre-registered decision rule (Stage D): if the
low-vs-deep difference of the subspace projections exceeds the even-vs-odd control, δ̂ is
partly absorbing compression-dependent model error rather than measuring the machine.

  python scripts/stageD2_metrology_report.py --arm A1_B --tag _metro --lam 1e-2
-> results/stageD2/15673/metrology_<arm><tag>.json (+ a printed summary)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results" / "stageD2" / "15673"


def coords(r, names, sig):
    """Run -> vector in prior-σ units, ordered as the registration's param_names."""
    v = []
    for n in names:
        x = (r["l1_shift_deg"] if n == "l1_shift" else r["l2_shift_deg"] if n == "l2_shift"
             else r["l2_gain"] - 1.0 if n == "l2_gain" else r["delta_phys"][n])
        v.append(x / sig[n])
    return np.array(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default="_metro")
    ap.add_argument("--lam", default="1e-2", help="λ of the theft control used for recovery factors")
    ap.add_argument("--lev-min", type=float, default=0.2, help="leverage cut for the knob table")
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO / "configs/fit_stageD2_15673.yaml").read_text())
    base = a.arm + a.tag
    reg = json.loads((OUT / f"register_{base}_r0.json").read_text())
    names, V = reg["param_names"], np.array(reg["kept_directions"])
    sig = {k: float(v) for k, v in cfg["sigma_phys"].items()}
    sig.update(l1_shift=float(cfg["l1ref_sigma_deg"]), l2_shift=float(reg["prior_scale"]["l2_shift"]),
               l2_gain=float(reg["prior_scale"].get("l2_gain", 0.25)))

    restarts = []
    for f in sorted((OUT / "reach").glob(f"{base}_r*.json")):
        for r in json.loads(f.read_text())["runs"]:
            restarts.append((f.stem, r))
    rep = {"arm": a.arm, "tag": a.tag, "n_kept": reg["n_kept"], "restarts": [], "splits": {}}
    P = []
    for nm, r in restarts:
        x = coords(r, names, sig)
        P.append(V @ x)
        rep["restarts"].append({"file": nm, "seed": r["seed"], "img": r["final_img_loss"],
                                "total": r["total_loss"], "walls": r["walls"],
                                "l2_gain": r.get("l2_gain"), "proj": (V @ x).tolist(),
                                "pulls": dict(zip(names, x.tolist()))})
    P = np.array(P)
    if len(P) > 1:
        rep["restart_proj_spread_max"] = float((P.max(0) - P.min(0)).max())
        X = np.array([list(r["pulls"].values()) for r in rep["restarts"]])
        rep["restart_pull_spread"] = dict(zip(names, (X.max(0) - X.min(0)).tolist()))
    best = min(rep["restarts"], key=lambda r: r["total"]) if rep["restarts"] else None

    for sp in ("even", "odd", "low", "deep"):
        f = OUT / "reach" / f"{base}_fit_{sp}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())["runs"][0]
        x = coords(r, names, sig)
        rep["splits"][sp] = {"img": r["final_img_loss"], "holdout": r.get("holdout"),
                             "l2_gain": r.get("l2_gain"), "proj": (V @ x).tolist(),
                             "pulls": dict(zip(names, x.tolist()))}
    S = rep["splits"]
    for p, q in (("even", "odd"), ("low", "deep")):
        if p in S and q in S:
            d = np.array(S[p]["proj"]) - np.array(S[q]["proj"])
            dx = np.array(list(S[p]["pulls"].values())) - np.array(list(S[q]["pulls"].values()))
            S[f"{p}_vs_{q}"] = {"proj_diff_rms": float(np.sqrt((d ** 2).mean())),
                                "proj_diff_max": float(np.abs(d).max()),
                                "pull_diff_rms": float(np.sqrt((dx ** 2).mean()))}
    if "even_vs_odd" in S and "low_vs_deep" in S:
        rep["decision"] = ("compression-dependent (low/deep > even/odd control)"
                           if S["low_vs_deep"]["proj_diff_rms"] > S["even_vs_odd"]["proj_diff_rms"]
                           else "consistent (low/deep <= even/odd control)")

    th = OUT / "reach" / f"A1_B_theft_lam{a.lam}.json"
    if th.exists():
        tr = json.loads(th.read_text())["runs"][0]
        truth = json.loads((REPO / "data/s2e_theft_15673/points.json").read_text())[0]
        truth = {**truth["linac"], **truth["injector"]}
        rep["recovery_at_lam"] = {k: tr["delta_phys"][k] / v for k, v in truth.items()}
    lev = reg["leverage"]
    table = [n for n in names if lev[n] >= a.lev_min]
    rep["identifiable"] = table
    (OUT / f"metrology_{base}.json").write_text(json.dumps(rep, indent=1))

    print(f"{base}: {len(rep['restarts'])} restarts, {reg['n_kept']} registered directions")
    for r in rep["restarts"]:
        print(f"  {r['file']:28s} img {r['img']:.4f} total {r['total']:.4f} walls {r['walls']}"
              + (f" gain {r['l2_gain']:.3f}" if r.get("l2_gain") is not None else ""))
    if "restart_proj_spread_max" in rep:
        print(f"  restart subspace-projection spread (max over directions) "
              f"{rep['restart_proj_spread_max']:.3f} sigma")
    for k in ("even_vs_odd", "low_vs_deep"):
        if k in S:
            print(f"  {k}: subspace diff rms {S[k]['proj_diff_rms']:.3f} max {S[k]['proj_diff_max']:.3f}; "
                  f"all-pull diff rms {S[k]['pull_diff_rms']:.3f}")
    for sp in ("even", "odd", "low", "deep"):
        h = S.get(sp, {}).get("holdout")
        if h:
            print(f"  fit {sp:4s}: held-out img {h['img_held']:.4f} (null {h.get('null_held', float('nan')):.4f})"
                  f", fitted {h['img_fit']:.4f}")
    if "decision" in rep:
        print(f"  decision: {rep['decision']}")
    if best:
        print(f"  identifiable (leverage >= {a.lev_min}), best restart, pull in prior sigma:")
        for n in table:
            phys = best["pulls"][n] * sig[n]
            rec = rep.get("recovery_at_lam", {}).get(n)
            print(f"    {n:28s} lev {lev[n]:.2f}  pull {best['pulls'][n]:+6.2f}  value {phys:+.4g}"
                  + (f"  (S2E recovery {rec:.2f})" if rec is not None else ""))


if __name__ == "__main__":
    main()
