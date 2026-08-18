"""Stage-D D1: build the real E300_15673 per-(step,sign) target images.

Run under slac-rl (CPU; the DAQ stack + torch anchor op):

  cd /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning && \
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/build_targets_15673.py

Outputs:
  results/stageD/15673/targets.npz   (images, off_images, mask, splits,
                                      floors, shot counts, provenance)
  results/stageD/15673/qa/step_XX.png  target / off / mask overlays
  results/stageD/15673/targets_qa.json summary numbers
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xtcav_gpsr_tuning.targets_real import build_targets  # noqa: E402

DATASET = "E300_15673"
OUT = REPO / "results" / "stageD" / "15673"


def main():
    out = OUT
    (out / "qa").mkdir(parents=True, exist_ok=True)
    res = build_targets(DATASET)

    np.savez_compressed(out / "targets.npz", **res)
    print(f"targets: {out / 'targets.npz'}")

    imgs, off_imgs, mask = res["images"], res["off_images"], res["mask"]
    n_steps = imgs.shape[0]
    ext = None
    for si in range(n_steps):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
        for gi, sign in enumerate(("plus", "neg")):
            ax = axes[gi]
            im = ax.imshow(imgs[si, gi], origin="lower", aspect="auto",
                           cmap="viridis", extent=ext)
            ax.set_title(f"step {res['steps'][si]} {sign} "
                         f"(n={res['n_shots'][si, gi]}, "
                         f"floor {res['floors_tv'][si, gi]:.3f})", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ys, xs = np.where(mask < 0.5)
            if len(ys):
                ax.plot(xs, ys, "rx", ms=3)
        ax = axes[2]
        im = ax.imshow(off_imgs[si], origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"off (pooled n={res['n_off'][si]})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{DATASET} scan {res['scan_vals_deg'][si]:+.2f} deg",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "qa" / f"step_{int(res['steps'][si]):02d}.png", dpi=110)
        plt.close(fig)

    qa = {
        "dataset": DATASET,
        "n_steps": int(n_steps),
        "n_shots_median": float(np.median(res["n_shots"])),
        "n_shots_min": int(res["n_shots"].min()),
        "empty_targets": int((res["n_shots"] == 0).sum()),
        "floor_tv_median": float(np.nanmedian(res["floors_tv"])),
        "floor_tv_max": float(np.nanmax(res["floors_tv"])),
        "split_median_plus": float(np.nanmedian(res["splits"][:, 0])),
        "split_median_neg": float(np.nanmedian(res["splits"][:, 1])),
        "widened_targets": int((res["phase_window_used_deg"] > 0.21).sum()),
        "n_masked_px": int((mask < 0.5).sum()),
        "provenance": json.loads(str(res["provenance"])),
    }
    (out / "targets_qa.json").write_text(json.dumps(qa, indent=1))
    print(json.dumps(qa, indent=1))


if __name__ == "__main__":
    main()
