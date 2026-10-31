"""Stage-D2: macroparticle-count check. True S2E at the A1_B fitted state with 1e5
(s2e_csrbin32_15673: the 40-bin reference) and 1e6 macroparticles (s2e_np1e6_15673, fresh
Impact-T + Bmad, same 40 CSR bins) at the steps both probes share, next to the real image.

Per step and streak sign: TV 1e6 vs 1e5 and vs real, drive band width (stageD2_thickness), and
the best global Gaussian blur vs real (grid over σ_E, σ_t in px, renormalized after the blur).

  python scripts/stageD2_np1e6_figure.py
-> results/stageD2/15673/figures/np1e6_plus.png, results/stageD2/15673/np1e6.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from stageD2_figures import VMAX_PCT, targets  # noqa: E402
from stageD2_thickness import band_width  # noqa: E402

PROBES = {"1e5": "s2e_csrbin32_15673", "1e6": "s2e_np1e6_15673"}
GRID = np.arange(0, 9)


def load(probe):
    d = REPO / "data" / probe
    return {r["step"]: np.load(d / f"render_B_{r['idx']:05d}.npz")
            for r in json.loads((d / "manifest.json").read_text())}


def best_blur(im, ref):
    best = (np.abs(im - ref).sum(), 0, 0)
    for se in GRID:
        for st in GRID:
            b = gaussian_filter(im, (se, st), mode="constant")
            tv = np.abs(b / b.sum() - ref).sum()
            if tv < best[0]:
                best = (tv, int(se), int(st))
    return {"tv": float(best[0]), "sigma_E_px": best[1], "sigma_t_px": best[2]}


def global_blur(imgs, real, steps):
    """One blur for every (step, sign) image, as the CSR-bin check: mean TV vs real."""
    pairs = [(s, gi, sg) for s in steps for gi, sg in enumerate(("plus", "minus"))]
    tv = lambda se, st: float(np.mean([np.abs(b / b.sum() - real[gi, s]).sum() for s, gi, sg in pairs
                                       for b in [gaussian_filter(imgs[s][sg], (se, st), mode="constant")]]))
    grid = {(int(se), int(st)): tv(se, st) for se in GRID for st in GRID}
    (se, st), best = min(grid.items(), key=lambda kv: kv[1])
    return {"tv_unblurred": grid[(0, 0)], "tv": best, "sigma_E_px": se, "sigma_t_px": st}


def main():
    real = targets("B")["images"].numpy()
    imgs = {k: load(p) for k, p in PROBES.items()}
    steps = sorted(set(imgs["1e5"]) & set(imgs["1e6"]))
    out = {}
    for s in steps:
        rec = {}
        for gi, sign in enumerate(("plus", "minus")):
            r = real[gi, s]
            a, b = imgs["1e5"][s][sign], imgs["1e6"][s][sign]
            rec[sign] = {"tv_1e6_vs_1e5": float(np.abs(a - b).sum()),
                         "tv_vs_real": {"1e5": float(np.abs(a - r).sum()), "1e6": float(np.abs(b - r).sum())},
                         "drive_width_um": {"real": band_width(r, "drive"), "1e5": band_width(a, "drive"),
                                            "1e6": band_width(b, "drive")},
                         "best_blur": {"1e5": best_blur(a, r), "1e6": best_blur(b, r)}}
        a, b = imgs["1e5"][s]["off"], imgs["1e6"][s]["off"]
        rec["off_tv_1e6_vs_1e5"] = float(np.abs(a - b).sum())
        out[int(s)] = rec
        p = rec["plus"]
        print(f"step {s}: TV 1e6-1e5 {p['tv_1e6_vs_1e5']:.3f} / minus {rec['minus']['tv_1e6_vs_1e5']:.3f} "
              f"(off {rec['off_tv_1e6_vs_1e5']:.3f}); vs real 1e5 {p['tv_vs_real']['1e5']:.3f} 1e6 "
              f"{p['tv_vs_real']['1e6']:.3f}; drive width real/1e5/1e6 "
              f"{p['drive_width_um']['real']:.0f}/{p['drive_width_um']['1e5']:.0f}/{p['drive_width_um']['1e6']:.0f} um; "
              f"best blur 1e5 {p['best_blur']['1e5']} 1e6 {p['best_blur']['1e6']}", flush=True)
    out["global_blur"] = {k: global_blur(imgs[k], real, steps) for k in PROBES}
    print("global blur:", out["global_blur"], flush=True)
    (REPO / "results/stageD2/15673/np1e6.json").write_text(json.dumps(out, indent=1))

    fig, axes = plt.subplots(len(steps), 3, figsize=(7.2, 1.9 * len(steps)), squeeze=False)
    for i, s in enumerate(steps):
        for j, c in enumerate(["real", "1e5", "1e6"]):
            im = real[0, s] if c == "real" else imgs[c][s]["plus"]
            ax = axes[i, j]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0,
                      vmax=np.percentile(im, VMAX_PCT))
            ax.set_xticks([]); ax.set_yticks([])
            if c != "real":
                txt = f"vs real {out[s]['plus']['tv_vs_real'][c]:.2f}"
                if c == "1e6":
                    txt += f"\nvs 1e5 {out[s]['plus']['tv_1e6_vs_1e5']:.2f}"
                ax.text(0.98, 0.04, txt, color="w", fontsize=6.5, ha="right", va="bottom",
                        transform=ax.transAxes)
            if i == 0:
                ax.set_title("real" if c == "real" else f"S2E, {c} macroparticles", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"scan {0.25 * s:+.2f}°", fontsize=8)
    fig.suptitle("Macroparticle count at the A1_B fitted state (40 CSR bins), DTOTR2 plus, convention B",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = REPO / "results/stageD2/15673/figures/np1e6_plus.png"
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
