"""Why are the model blobs diffuse? Decompose rendered feature width into
flow-intrinsic width vs KDE kernel, against the real target's core widths.

Renders the fitted flip1_swap1 model at one mid-scan step with a ladder of KDE
bandwidths and measures, identically on model and real images:
  * y-FWHM through the peak column (energy-axis core thickness)
  * x-FWHM through the peak row (streak-axis feature width)

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=$PWD/src:.../twobunch-s2e-rl/src python scripts/stageD_blur_decomposition.py
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
from stageD_figures import set_model_params  # noqa: E402
from xtcav_gpsr_tuning.screen import PIXEL_UM, ScreenGrid  # noqa: E402
from xtcav_gpsr_tuning.stage_d import (conditioning, load_stage_d,  # noqa: E402
                                       load_targets)

FIT = REPO / "results" / "stageD" / "15673" / "fit"
OUT = REPO / "results" / "stageD" / "15673" / "figures"
STEP = 16  # scan +4.00 deg
BW_UM = [30.5, 61.0, 122.0, 185.0, 264.0]


def fwhm_px(prof):
    p = prof - prof.min()
    half = p.max() / 2.0
    above = np.where(p >= half)[0]
    return float(above[-1] - above[0] + 1) if len(above) else np.nan


def core_widths(img):
    """(y-FWHM through peak column, x-FWHM through peak row), in µm."""
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    return fwhm_px(img[:, ix]) * PIXEL_UM, fwhm_px(img[iy, :]) * PIXEL_UM


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_stage_d(device=device)
    base = conditioning(setup, cfg)
    real = load_targets(cfg, device=device)
    ens = json.loads((FIT / "ensemble.json").read_text())
    runs = ens["hypotheses"]["flip1_swap1"]["runs"]
    best = min([r for r in runs if r["final_img_loss"] < 1.0],
               key=lambda r: r["final_img_loss"])
    tgt = orient_targets(real, flip_rows=True, swap_signs=True)
    real_img = tgt["images"][0, STEP].cpu().numpy()

    model = make_model(setup, cfg, base, seed=0)
    set_model_params(model, best)

    imgs, widths = {}, {}
    for bw in BW_UM:
        model.s.grid = ScreenGrid(bandwidth_m=bw * 1e-6, device=device)
        with torch.no_grad():
            im, _ = model(rows=[STEP])
        imgs[bw] = im[0, 0].cpu().numpy()
        widths[bw] = core_widths(imgs[bw])
    w_real = core_widths(real_img)

    fig, axes = plt.subplots(1, len(BW_UM) + 1, figsize=(3 * (len(BW_UM) + 1), 3.2))
    ax = axes[0]
    ax.imshow(real_img, origin="lower", aspect="auto", cmap="viridis")
    ax.set_title(f"REAL\ncore y-FWHM {w_real[0]:.0f} µm\nx-FWHM {w_real[1]:.0f} µm",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for ax, bw in zip(axes[1:], BW_UM):
        ax.imshow(imgs[bw], origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"model bw={bw:.0f} µm\ny-FWHM {widths[bw][0]:.0f} µm\n"
                     f"x-FWHM {widths[bw][1]:.0f} µm", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"E300_15673 step scan +4.00° (+ sign): real vs fitted model "
                 f"across KDE bandwidth (n={model.n} per bunch)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "blur_decomposition.png", dpi=130)

    out = {"step": STEP, "real_yfwhm_um": w_real[0], "real_xfwhm_um": w_real[1],
           "model": {str(bw): {"yfwhm_um": widths[bw][0],
                               "xfwhm_um": widths[bw][1]} for bw in BW_UM}}
    (OUT / "blur_decomposition.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
