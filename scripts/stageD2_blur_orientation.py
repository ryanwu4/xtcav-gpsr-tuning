"""Stage-D2: does the resolution term's axis split follow the trace orientation?

An axis-aligned Gaussian blur only widens a trace across itself: a trace running along time is
widened by an energy blur, one running along energy by a time blur. If the real traces are
thicker PERPENDICULAR to themselves (a thicker longitudinal phase space), the best axis-aligned
blur should swing from time to energy as the traces turn, which is what the low / deep split
fits show (pure time blur on the low half, time + energy on the deep half).

True S2E at the A1_B fitted state (s2e_probe_15673_v3_heater_delay A1B_fit: steps 2/10/16/26;
s2e_csrbin32_15673: 29/32, the same 40-bin S2E), per step and streak sign:
  - trace orientation: share of the S2E image's squared gradient along the energy axis,
    f_E = <g_E^2> / (<g_E^2> + <g_t^2>) (structure tensor, 1 px Gaussian pre-smoothing):
    f_E -> 1 for traces running along time, -> 0 for traces running along energy;
  - best axis-aligned blur of S2E vs real (grid σ_E, σ_t = 0..8 px, renormalized), and the
    best single-σ blur ACROSS the local trace (σ_perp; σ_E = σ_perp sqrt(f_E),
    σ_t = σ_perp sqrt(1 - f_E), an approximation for straight traces).

  PYTHONPATH=$PWD/src python scripts/stageD2_blur_orientation.py
-> results/stageD2/15673/blur_orientation.json (+ a printed table)
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from stageD2_figures import targets  # noqa: E402

SOURCES = [("s2e_probe_15673_v3_heater_delay", "A1B_fit"), ("s2e_csrbin32_15673", "A1B_fit")]
GRID = np.arange(0, 9)


def renders():
    out = {}
    for probe, point in SOURCES:
        d = REPO / "data" / probe
        for r in json.loads((d / "manifest.json").read_text()):
            if r["injector_point"] == point and r["step"] not in out:
                out[r["step"]] = np.load(d / f"render_B_{r['idx']:05d}.npz")
    return dict(sorted(out.items()))


def energy_gradient_share(im):
    s = gaussian_filter(im, 1.0)
    ge, gt = np.gradient(s)            # axis 0 = energy (rows), axis 1 = time (columns)
    return float((ge ** 2).sum() / ((ge ** 2).sum() + (gt ** 2).sum()))


def tv_blur(im, ref, se, st):
    b = gaussian_filter(im, (se, st), mode="constant")
    return float(np.abs(b / b.sum() - ref).sum())


def main():
    real = targets("B")["images"].numpy()
    imgs = renders()
    rows, out = [], {}
    for s, z in imgs.items():
        for gi, sign in enumerate(("plus", "minus")):
            im, ref = z[sign], real[gi, s]
            fE = energy_gradient_share(im)
            grid = {(int(a), int(b)): tv_blur(im, ref, a, b) for a in GRID for b in GRID}
            (se, st), tv = min(grid.items(), key=lambda kv: kv[1])
            perp = {float(p): tv_blur(im, ref, p * np.sqrt(fE), p * np.sqrt(1 - fE))
                    for p in np.arange(0, 10.5, 0.5)}
            sp, tvp = min(perp.items(), key=lambda kv: kv[1])
            rec = {"f_E": fE, "tv0": grid[(0, 0)], "axis": {"sigma_E": se, "sigma_t": st, "tv": tv},
                   "perp": {"sigma": sp, "tv": tvp}, "f_E_real": energy_gradient_share(ref)}
            out.setdefault(int(s), {})[sign] = rec
            rows.append((fE, se, st, tv, sp, tvp, grid[(0, 0)]))
            print(f"step {s:2d} {sign:5s} f_E S2E {fE:.2f} real {rec['f_E_real']:.2f}  TV {grid[(0, 0)]:.3f} -> axis "
                  f"σ_E {se} σ_t {st} {tv:.3f} | perp σ {sp:.1f} {tvp:.3f}", flush=True)
    r = np.array(rows)
    share = r[:, 1] ** 2 / np.maximum(r[:, 1] ** 2 + r[:, 2] ** 2, 1e-9)
    ok = (r[:, 1] + r[:, 2]) > 0
    out["summary"] = {"corr_fE_vs_energy_blur_share": float(np.corrcoef(r[ok, 0], share[ok])[0, 1]),
                      "mean_tv0": float(r[:, 6].mean()), "mean_tv_axis": float(r[:, 3].mean()),
                      "mean_tv_perp": float(r[:, 5].mean()),
                      "perp_sigma_median": float(np.median(r[:, 4])),
                      "perp_sigma_spread": [float(r[:, 4].min()), float(r[:, 4].max())]}
    print("summary:", out["summary"])
    path = REPO / "results/stageD2/15673/blur_orientation.json"
    path.write_text(json.dumps(out, indent=1))
    print(path)


if __name__ == "__main__":
    main()
