"""Stage-D2: DTOTR2 image gallery for the quad-field-hold probe (s2e_l2hold_15673).

Per probe step (rows): the real 15673 image, one machine-like geometry re-LEM'd every step
(s2e_l2machine_15673, S2E's default), the same rows with the quad fields held at the scan centre
and at the scan start, and each held version minus the re-LEM'd one. Image panels are scaled to
their own 99.5th pct and labelled with TV vs real; difference panels share one symmetric scale per
row and are labelled with their TV.

  python scripts/stageD2_l2hold_gallery.py --geometry l2m_A --sign plus
-> results/stageD2/15673/figures/l2hold_gallery_{geometry}_{sign}.png
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

NEW, LEM = REPO / "data/s2e_l2hold_15673", REPO / "data/s2e_l2machine_15673"
COLS = ["real", "re-LEM'd", "held @ centre", "centre − re-LEM'd", "held @ start", "start − re-LEM'd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="l2m_A", choices=["l2m_A", "l2m_B"])
    ap.add_argument("--sign", default="plus", choices=["plus", "minus", "off"])
    a = ap.parse_args()
    real = targets("B")["images"].numpy()
    idx = lambda d: {(r["injector_point"], r["step"]): r["idx"]
                     for r in json.loads((d / "manifest.json").read_text())}
    new, lem = idx(NEW), idx(LEM)
    steps = sorted(s for p, s in lem if p == a.geometry)
    render = lambda d, i: np.load(d / f"render_B_{i:05d}.npz")[a.sign]
    fig, axes = plt.subplots(len(steps), len(COLS), figsize=(1.9 * len(COLS), 1.35 * len(steps)),
                             squeeze=False)
    for i, si in enumerate(steps):
        im = {"re-LEM'd": render(LEM, lem[(a.geometry, si)]),
              "held @ centre": render(NEW, new[(a.geometry + "_lemC", si)]),
              "held @ start": render(NEW, new[(a.geometry + "_lem0", si)])}
        if a.sign != "off":
            im["real"] = real[0 if a.sign == "plus" else 1, si]
        d = {"centre − re-LEM'd": im["held @ centre"] - im["re-LEM'd"],
             "start − re-LEM'd": im["held @ start"] - im["re-LEM'd"]}
        dmax = max(max(np.percentile(np.abs(x), VMAX_PCT) for x in d.values()), 1e-12)
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
                ax.set_ylabel(f"scan {0.25 * si:+.2f}°", fontsize=8)
    label = {"l2m_A": "A: LI11 in phase with LI12-14", "l2m_B": "B: LI11 on crest"}[a.geometry]
    fig.suptitle(f"Machine-like L2 scan in S2E, quad fields held (no re-LEM) vs re-LEM'd every step; "
                 f"{label}; DTOTR2 {a.sign}, convention B", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = REPO / f"results/stageD2/15673/figures/l2hold_gallery_{a.geometry}_{a.sign}.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
