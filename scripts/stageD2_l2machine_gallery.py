"""Stage-D2: DTOTR2 image gallery for the machine-like L2 scan probe (s2e_l2machine_15673).

Per probe step (rows): the real 15673 image, the S2E render of the standard model scan of the same
state (s2e_theft_15673), and each machine-like geometry (l2m_A: LI11 in phase, l2m_B: LI11 on
crest) with two difference panels: point − model scan (the whole effect) and point − its _iso twin
(the energy-profile part). Image panels are scaled to their own 99.5th pct and labelled with TV vs
real; difference panels share one symmetric scale per row and are labelled with their TV.

  python scripts/stageD2_l2machine_gallery.py --sign plus
-> results/stageD2/15673/figures/l2machine_gallery_{sign}.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from stageD2_figures import VMAX_PCT, targets  # noqa: E402

NEW, REF = REPO / "data/s2e_l2machine_15673", REPO / "data/s2e_theft_15673"
COLS = ["real", "model scan", "l2m_A", "A − scan", "A − iso", "l2m_B", "B − scan", "B − iso"]
DIFF = {"A − scan": ("l2m_A", "model scan"), "A − iso": ("l2m_A", "l2m_A_iso"),
        "B − scan": ("l2m_B", "model scan"), "B − iso": ("l2m_B", "l2m_B_iso")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", default="plus", choices=["plus", "minus", "off"])
    a = ap.parse_args()
    real = targets("B")["images"].numpy()
    scan = np.asarray(json.loads(Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results/"
                                      "tier1_sim_scans/E300_15673/machine_config.json")
                                 .read_text())["L2_scan_vals_deg"])
    ref = {r["step"]: r["idx"] for r in json.loads((REF / "manifest.json").read_text())}
    new = {(r["injector_point"], r["step"]): r["idx"]
           for r in json.loads((NEW / "manifest.json").read_text())}
    steps = sorted(ref)
    render = lambda d, i: np.load(d / f"render_B_{i:05d}.npz")[a.sign]
    fig, axes = plt.subplots(len(steps), len(COLS), figsize=(1.9 * len(COLS), 1.35 * len(steps)),
                             squeeze=False)
    for i, si in enumerate(steps):
        im = {"model scan": render(REF, ref[si])}
        for p in ("l2m_A", "l2m_A_iso", "l2m_B", "l2m_B_iso"):
            im[p] = render(NEW, new[(p, si)])
        if a.sign != "off":
            im["real"] = real[0 if a.sign == "plus" else 1, si]
        d = {c: im[p] - im[q] for c, (p, q) in DIFF.items()}
        dmax = max(np.percentile(np.abs(x), VMAX_PCT) for x in d.values())
        for j, c in enumerate(COLS):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if c in d:
                ax.imshow(d[c], origin="lower", aspect="auto", cmap="RdBu_r", vmin=-dmax, vmax=dmax)
                ax.text(0.98, 0.04, f"TV {np.abs(d[c]).sum():.2f}", color="k", fontsize=7,
                        ha="right", va="bottom", transform=ax.transAxes)
            elif c in im:
                ax.imshow(im[c], origin="lower", aspect="auto", cmap="viridis", vmin=0,
                          vmax=np.percentile(im[c], VMAX_PCT))
                if c != "real" and "real" in im:
                    ax.text(0.98, 0.04, f"vs real {np.abs(im[c] - im['real']).sum():.2f}",
                            color="w", fontsize=7, ha="right", va="bottom", transform=ax.transAxes)
            else:
                ax.axis("off")
            if i == 0:
                ax.set_title(c, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"scan {scan[si]:+.2f}°", fontsize=8)
    fig.suptitle(f"Machine-like L2 scan in S2E, DTOTR2 {a.sign} streak (convention B). A: LI11 in "
                 f"phase with LI12-14; B: LI11 on crest. Differences share one scale per row.",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = REPO / f"results/stageD2/15673/figures/l2machine_gallery_{a.sign}.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
