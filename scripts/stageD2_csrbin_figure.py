"""Stage-D2: CSR-binning convergence gallery. True S2E at the A1_B fitted state with the 1D CSR
calculation at 40 (FACET2-S2E's Activate_CSR.tao default, s2e_csrbin32_15673: its csr_n_bin never
reached Tao), 80, 160 and 320 longitudinal bins, next to the real image; per row one probe step,
each panel scaled to its own 99.5th pct, labelled with TV vs real and vs the next-finer binning.

  python scripts/stageD2_csrbin_figure.py --sign plus
-> results/stageD2/15673/figures/csrbin_{sign}.png
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

PROBES = {40: "s2e_csrbin32_15673", 80: "s2e_csrbin80_15673", 160: "s2e_csrbin160_15673",
          320: "s2e_csrbin320_15673"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", default="plus", choices=["plus", "minus"])
    a = ap.parse_args()
    gi = 0 if a.sign == "plus" else 1
    real = targets("B")["images"].numpy()
    imgs = {}
    for nb, p in PROBES.items():
        d = REPO / "data" / p
        for r in json.loads((d / "manifest.json").read_text()):
            imgs[(nb, r["step"])] = np.load(d / f"render_B_{r['idx']:05d}.npz")[a.sign]
    steps = sorted({s for _, s in imgs})
    bins = list(PROBES)
    fig, axes = plt.subplots(len(steps), 1 + len(bins), figsize=(2.3 * (1 + len(bins)), 1.7 * len(steps)),
                             squeeze=False)
    for i, s in enumerate(steps):
        for j, c in enumerate(["real"] + bins):
            im = real[gi, s] if c == "real" else imgs[(c, s)]
            ax = axes[i, j]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0,
                      vmax=np.percentile(im, VMAX_PCT))
            ax.set_xticks([]); ax.set_yticks([])
            if c != "real":
                k = bins.index(c)
                txt = f"vs real {np.abs(im - real[gi, s]).sum():.2f}"
                if k + 1 < len(bins):
                    txt += f"\nvs {bins[k + 1]} bins {np.abs(im - imgs[(bins[k + 1], s)]).sum():.2f}"
                ax.text(0.98, 0.04, txt, color="w", fontsize=6.5, ha="right", va="bottom",
                        transform=ax.transAxes)
            if i == 0:
                ax.set_title("real" if c == "real" else f"S2E, {c} CSR bins", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"scan {0.25 * s:+.2f}°", fontsize=8)
    fig.suptitle(f"CSR binning convergence at the A1_B fitted state (1e5 macroparticles), DTOTR2 "
                 f"{a.sign}, convention B", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = REPO / f"results/stageD2/15673/figures/csrbin_{a.sign}.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
