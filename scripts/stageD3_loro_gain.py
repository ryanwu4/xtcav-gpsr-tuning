"""Stage D3: leave-one-run-out transfer with the held-out run's own L2 scan gain.

The LORO<run> arms fit the other two runs' windows, then refit only the held-out run's L1 and L2
phases. The scan gain turned out to differ between runs (15668 ~1, 15671/15673 ~0.77), so this
re-uses each LORO<run> shared state and refits the held-out run's L1, L2 AND scan gain on its
window, then scores its whole scan.

  PYTHONPATH=$PWD/src ~/miniconda3/envs/slac-rl/bin/python scripts/stageD3_loro_gain.py 15668 15671 15673
-> results/stageD3/fits/LOROg<run>.json, results/stageD3/render/LOROg<run>.npz
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_stageD3_fit as R  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for held in sys.argv[1:]:
        t0 = time.time()
        src = json.loads((R.OUT / "fits" / f"LORO{held}.json").read_text())
        best = min(src["fits"], key=lambda x: x["final_img_loss"])
        runs, drift = src["runs"], src["drift"]
        cfg, setup, base, ridx, scan, tgt, meta = R.build((held,), "win", drift, device)
        m, cfg_d2 = R.make_model(cfg, setup, base, ridx, scan, device)
        R.load_state(m, best["state"], list(runs), [held])
        for n, p in m.named_parameters():
            p.requires_grad_(n in ("l1_shift_deg", "l2_shift_deg", "l2_gain_m1"))
        null = R.evaluate(m, tgt, cfg)
        res = R.fit(m, tgt, cfg, cfg_d2, max(40, R.ITERS // 2))
        res.pop("loss_history")
        parts = R.evaluate(m, tgt, cfg)
        res.update(img_before_refit=null["img"], final_img_loss=parts["img"],
                   per_image_tv=parts["per_image_tv"].tolist(), meta=meta, state=R.state(m))
        R.summarize(m, res, (held,))
        sc, imgs = R.score_full(res["state"], (held,), drift, device)
        out = {"arm": f"LOROg{held}", "runs": runs, "held_out": held, "drift": drift,
               "source_arm": f"LORO{held}", "fits": src["fits"], "held_out_fit": res,
               "score_full": {**sc, "seed": best["seed"]}}
        (R.OUT / "fits" / f"LOROg{held}.json").write_text(json.dumps(out, indent=1))
        np.savez_compressed(R.OUT / "render" / f"LOROg{held}.npz", imgs=imgs.astype(np.float32),
                            meta=np.array(json.dumps(sc["meta"])))
        tv = np.asarray(sc["per_image_tv"]).mean(0)
        print(f"LOROg{held}: window img {null['img']:.4f} -> {parts['img']:.4f} after refitting L1, L2 and "
              f"gain; g {res['per_run'][held]['l2_gain']:.3f}, L1 {res['per_run'][held]['l1_deg']:.2f} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        print(f"  full-scan TV {held}: " + " ".join(f"{x:.2f}" for x in tv), flush=True)


if __name__ == "__main__":
    main()
