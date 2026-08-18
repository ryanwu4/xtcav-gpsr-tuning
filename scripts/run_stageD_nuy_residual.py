"""D0.5 follow-up: nu_y contamination from RESIDUAL jitter after the ±0.2°
phase cut (uniform in the window), at the real geometry. Decides phi_ensemble.

  cd /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning && \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src:/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/run_stageD_nuy_residual.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from run_stageD_precal import make_truth, one_fit  # noqa: E402
from xtcav_gpsr_tuning.stage_d import conditioning, load_stage_d  # noqa: E402

OUT = REPO / "results" / "stageD" / "15673" / "precal"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_stage_d(device=device)
    base = conditioning(setup, cfg)
    n_p, lam = 8192, 3e-3
    rng = np.random.default_rng(11)
    runs = []
    for s in range(5):
        imgs_acc, splits_acc = None, None
        for _ in range(8):
            phi = float(rng.uniform(-0.2, 0.2))  # residual jitter after the cut
            im, sp = make_truth(setup, cfg, base, {}, seed=int(rng.integers(1 << 30)),
                                n_particles=n_p // 2, phi_deg=phi)
            imgs_acc = im if imgs_acc is None else imgs_acc + im
            splits_acc = sp if splits_acc is None else splits_acc + sp
        r = one_fit(setup, cfg, base, (imgs_acc / 8.0, splits_acc / 8.0), lam, s,
                    n_p, iters=150)
        runs.append({"seed": s, "nu_y": r["nu_y"],
                     "img_loss": r["final_img_loss"],
                     "max_pull": max(abs(v) for v in r["pulls"].values())})
        print(f"seed {s}: nu_y {r['nu_y']:.4f} img {r['final_img_loss']:.4f}",
              flush=True)
    nus = [r["nu_y"] for r in runs]
    out = {"note": "residual jitter after the +/-0.2 deg phase cut "
                   "(uniform in window); decision: phi_ensemble stays null "
                   "if |median-1| < 0.02",
           "runs": runs, "nu_y_median": float(np.median(nus)),
           "nu_y_spread": float(np.ptp(nus))}
    (OUT / "nuy_residual.json").write_text(json.dumps(out, indent=1))
    print("median", round(out["nu_y_median"], 4),
          "spread", round(out["nu_y_spread"], 4))


if __name__ == "__main__":
    main()
