"""Stage-D2: targets npz (load_targets schema) from an S2E probe's renders, for sim-to-sim
fits against TRUE Impact-T + Bmad images (the Phase-2 theft control).

Renders come from score_s2e_probe.py (render_<conv>_<idx>.npz, model orientation). They are
stored here in the CANONICAL target frame (rows flipped, signs swapped), so the fit's
flip1_swap1 orient_targets maps them back to the model frame exactly. Steps without a probe
row get zero images and NaN floors -> zero step weight (excluded from the fit). Floors of the
probed steps = the real targets' floors (the same noise weighting as the real fit).

  python scripts/build_s2e_targets.py --probe s2e_theft_15673 --conv B
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning.screen import image_zlobe_split  # noqa: E402

REAL = {"B": "results/stageD2/15673/targets_rowref_clean.npz",
        "N": "results/stageD2/15673/targets_rowref_none.npz",
        "D": "results/stageD/15673/targets.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--conv", default="B")
    a = ap.parse_args()
    pdir = REPO / "data" / a.probe
    real = np.load(REPO / REAL[a.conv])
    S, _, NY, NX = real["images"].shape
    imgs = np.zeros((S, 2, NY, NX)); off = np.zeros((S, NY, NX))
    floors = np.full((S, 2), np.nan); off_floors = np.full(S, np.nan)
    for r in json.loads((pdir / "manifest.json").read_text()):
        f = pdir / f"render_{a.conv}_{r['idx']:05d}.npz"
        if not f.exists():
            continue
        d, si = np.load(f), r["step"]
        # model frame (+V, -V) -> canonical frame: swap signs, flip rows (orient is an involution)
        imgs[si, 0], imgs[si, 1] = d["minus"][::-1], d["plus"][::-1]
        off[si] = d["off"][::-1]
        floors[si], off_floors[si] = real["floors_tv"][si], real["off_floors_tv"][si]
    splits = image_zlobe_split(torch.tensor(imgs)).numpy()
    out = {k: real[k] for k in ("mask", "steps", "scan_vals_deg")}
    out.update(images=imgs, off_images=off, splits=splits, floors_tv=floors,
               off_floors_tv=off_floors, n_shots=np.isfinite(floors).astype(int),
               provenance=np.array(json.dumps({"source": f"S2E probe {a.probe}", "conv": a.conv,
                                               "frame": "canonical (flip rows, swap signs)"})))
    path = REPO / "results/stageD2/15673" / f"targets_s2e_{a.probe}_{a.conv}.npz"
    np.savez_compressed(path, **out)
    print(path, "steps with targets:", int(np.isfinite(floors[:, 0]).sum()))


if __name__ == "__main__":
    main()
