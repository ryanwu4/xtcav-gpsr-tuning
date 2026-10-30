"""Stage-D2: re-render each reachability arm's best restart (and the δ=0 nulls) to
npz, so the gallery / feature figures run on CPU without refitting.

Best restart = the lowest MAP objective among restarts with image loss < CONV_TV_MAX
(Stage D's rule). The model is rebuilt with that restart's particle seed and its
saved parameter state, so the render is the fitted model exactly.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src python scripts/stageD2_render.py [arms...]

-> results/stageD2/15673/render/<arm>.npz  (imgs (2,B,ny,nx), off (B,ny,nx), splits)
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_stageD2_reach import ARMS, OUT, load_run, make_model  # noqa: E402
from run_stageD_fit import CONV_TV_MAX  # noqa: E402

RENDER = OUT.parent / "render"
NULLS = {f"null_{m}_{c}": f"{a}_{c}" for m, a in (("cb3", "A1"), ("xtcav23", "X1")) for c in "DNB"}


def best_restart(d: dict) -> dict:
    runs = d["runs"]
    conv = [r for r in runs if r["final_img_loss"] < CONV_TV_MAX] or runs
    return min(conv, key=lambda r: r["total_loss"])


@torch.no_grad()
def render(m) -> dict:
    B = m.base.shape[0]
    imgs, splits, offs = [], [], []
    for lo in range(0, B, 6):
        a, b, c = m.forward_full(rows=list(range(lo, min(lo + 6, B))))
        imgs.append(a.cpu()); splits.append(b.cpu()); offs.append(c.cpu())
    return {"imgs": torch.cat(imgs, 1).numpy(), "splits": torch.cat(splits, 1).numpy(),
            "off": torch.cat(offs, 0).numpy()}


def main():
    device = "cuda"
    RENDER.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or [p.stem for p in sorted(OUT.glob("*.json"))
                             if p.stem in ARMS] + list(NULLS)
    for name in names:
        if name in NULLS:
            m, _, _ = make_model(NULLS[name], device, seed=0)
            meta = {"arm": name, "seed": 0}
        else:
            d = json.loads((OUT / f"{name}.json").read_text())
            r = best_restart(d)
            m, _, _ = make_model(d["arm"], device, seed=r["seed"])
            load_run(m, r)
            meta = {"arm": name, "seed": r["seed"], "final_img_loss": r["final_img_loss"]}
        out = render(m)
        np.savez_compressed(RENDER / f"{name}.npz", meta=np.array(json.dumps(meta)), **out)
        print(f"rendered {name} -> {RENDER / f'{name}.npz'}", flush=True)


if __name__ == "__main__":
    main()
