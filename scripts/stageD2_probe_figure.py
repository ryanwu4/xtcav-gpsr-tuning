"""Stage-D2 S2E probe figure: per probe step, the real target, the S2E renders of every
probe point (score_s2e_probe.py npz), and optionally CNF renders (stageD2_render.py npz),
all under one row-correction convention. TV vs real in each panel.

  PYTHONPATH=$PWD/src python scripts/stageD2_probe_figure.py --probe s2e_probe_15673_v2 --conv B \
      --cnf A1_B --sign plus
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--conv", default="B")
    ap.add_argument("--cnf", nargs="*", default=[])
    ap.add_argument("--sign", default="plus", choices=["plus", "minus"])
    a = ap.parse_args()
    gi = 0 if a.sign == "plus" else 1
    pdir = REPO / "data" / a.probe
    rows = json.loads((pdir / "manifest.json").read_text())
    real = targets(a.conv)["images"].numpy()
    points = list(dict.fromkeys(r["injector_point"] for r in rows))
    steps = sorted({r["step"] for r in rows})
    cnf = {n: np.load(REPO / "results/stageD2/15673/render" / f"{n}.npz")["imgs"] for n in a.cnf}
    cols = ["real"] + points + list(cnf)
    scan = np.asarray(json.loads(Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results/"
                                      "tier1_sim_scans/E300_15673/machine_config.json")
                                 .read_text())["L2_scan_vals_deg"])
    fig, axes = plt.subplots(len(steps), len(cols), figsize=(2.3 * len(cols), 1.6 * len(steps)),
                             squeeze=False)
    for i, si in enumerate(steps):
        for j, c in enumerate(cols):
            if c == "real":
                im = real[gi, si]
            elif c in cnf:
                im = cnf[c][gi, si]
            else:
                r = next(r for r in rows if r["step"] == si and r["injector_point"] == c)
                f = pdir / f"render_{a.conv}_{r['idx']:05d}.npz"
                if not f.exists():
                    axes[i, j].axis("off")
                    continue
                im = np.load(f)[a.sign]
            ax = axes[i, j]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0,
                      vmax=np.percentile(im, VMAX_PCT))
            ax.set_xticks([]); ax.set_yticks([])
            if c != "real":
                ax.text(0.98, 0.04, f"TV {np.abs(im - real[gi, si]).sum():.2f}", color="w",
                        fontsize=7, ha="right", va="bottom", transform=ax.transAxes)
            if i == 0:
                ax.set_title(("S2E " if c not in cnf and c != "real" else "") + c, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"scan {scan[si]:+.2f}°", fontsize=8)
    fig.suptitle(f"{a.probe}: real vs S2E (Impact-T+Bmad, 1e5) vs CNF, {a.sign} sign, convention "
                 f"{a.conv}", fontsize=10)
    fig.tight_layout()
    out = REPO / "results/stageD2/15673/figures" / f"probe_{a.probe}_{a.conv}_{a.sign}.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
