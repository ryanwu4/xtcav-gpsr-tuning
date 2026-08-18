"""Kernel-policy comparison, fit-independent (valid regardless of any fit).

Renders the SAME δ=0 model at the v1 (264 µm) and v2 (1 px) kernels against the
real targets, with fair per-panel display scaling, and tabulates the width
diagnostics that decided the kernel policy. This is the honest "image quality"
comparison: no fitted parameters are involved on either side.

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=$PWD/src:.../twobunch-s2e-rl/src python scripts/stageD_kernel_compare.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from run_stageD_fit import make_model, orient_targets  # noqa: E402
from stageD_figures import VMAX_PCT, render_all  # noqa: E402
from xtcav_gpsr_tuning.screen import PIXEL_UM, ScreenGrid  # noqa: E402
from xtcav_gpsr_tuning.stage_d import (conditioning, load_stage_d,  # noqa: E402
                                       load_targets)

OUT = REPO / "results" / "stageD" / "15673" / "figures"
STEPS = [2, 16, 30]


def rms(img):
    x = np.arange(img.shape[-1], dtype=float)
    y = np.arange(img.shape[-2], dtype=float)
    px = img.sum(-2); py = img.sum(-1)
    px = px / px.sum(); py = py / py.sum()
    mx = (px * x).sum(); my = (py * y).sum()
    return (np.sqrt((px * (x - mx) ** 2).sum()) * PIXEL_UM,
            np.sqrt((py * (y - my) ** 2).sum()) * PIXEL_UM)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_stage_d(device=dev)
    base = conditioning(setup, cfg)
    real = load_targets(cfg, device=dev)
    tgt = orient_targets(real, True, True)   # canonical branch from both runs
    real_i = tgt["images"].cpu().numpy()
    real_off = tgt["off"].cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    scan = real["scan_vals_deg"]

    variants = {}
    for label, bw in (("v1 kernel 264 µm", 264.0), ("v2 kernel 1 px", 30.5)):
        setup.grid = ScreenGrid(bandwidth_m=bw * 1e-6, device=dev)
        m = make_model(setup, cfg, base, seed=0)   # δ = 0, pristine quads
        variants[label] = render_all(m)

    fig, axes = plt.subplots(3, len(STEPS), figsize=(13, 8))
    for c, si in enumerate(STEPS):
        rows = [("real target", real_i[0, si])] + \
               [(lab, v[0][0, si]) for lab, v in variants.items()]
        for r, (label, im) in enumerate(rows):
            ax = axes[r, c]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis",
                      vmin=0.0, vmax=np.percentile(im, VMAX_PCT))
            sx, sy = rms(im)
            ax.set_title(f"σx {sx:.0f} σy {sy:.0f} µm", fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0, c].set_title(f"scan {scan[si]:+.2f}°   " + axes[0, c].get_title(),
                             fontsize=9)
    fig.suptitle("Kernel policy at δ=0 (no fitted parameters): the 264 µm kernel "
                 "manufactured width the beam model does not have\n"
                 f"each panel scaled to its own {VMAX_PCT}th pct", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "kernel_compare.png", dpi=130)
    plt.close(fig)

    tab = {}
    for si in STEPS:
        tab[int(si)] = {
            "scan_deg": float(scan[si]),
            "real": {"streak": rms(real_i[0, si]), "off": rms(real_off[si])},
            **{lab: {"streak": rms(v[0][0, si]), "off": rms(v[1][si])}
               for lab, v in variants.items()},
        }
    (OUT / "kernel_compare.json").write_text(json.dumps(tab, indent=1, default=list))
    print(f"{'step':>5} {'chan':>7} | {'real x/y':>14} " +
          " ".join(f"{lab:>16}" for lab in variants))
    for si, r in tab.items():
        for chan in ("streak", "off"):
            cells = [f"{r['real'][chan][0]:6.0f}/{r['real'][chan][1]:5.0f}"] + \
                    [f"{r[lab][chan][0]:8.0f}/{r[lab][chan][1]:6.0f}" for lab in variants]
            print(f"{si:5d} {chan:>7} | " + " ".join(cells))
    print(f"figure -> {OUT / 'kernel_compare.png'}")


if __name__ == "__main__":
    main()
