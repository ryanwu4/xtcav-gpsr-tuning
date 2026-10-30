"""Stage-D2 retrain gate "residual aligns with injector": at a fitted reachability state,
how much of the remaining image residual (model − real, both signs + TCAV-off, step
weights as in the loss) lies in the span of the injector knobs' image responses, vs the
linac/FF knobs' and all knobs'?

Responses are central finite differences of the deterministic forward (fixed particle
seed and latent) per fitted knob, h = 0.02 of the normalized box, so the box walls do
not matter for the diagnostic (it asks what MORE of each direction would do). The L2
least-squares projection is a diagnostic of direction, not the TV loss itself; the
script also reports, per group, the TV after the best linear step (clipped at |step| ≤
the box span) as a first-order estimate of the extra gain.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src python scripts/stageD2_jacobian_alignment.py --arm A1_N
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_stageD2_reach import (ACTIVE6, ARMS, L0B2, LINAC17, OUT, PASSIVE6,  # noqa: E402
                               load_real, make_model)
from run_stageD_fit import orient_targets  # noqa: E402
from stageD2_render import best_restart  # noqa: E402

GROUPS = {"injector12": ACTIVE6 + PASSIVE6, "injector+L0B": ACTIVE6 + PASSIVE6 + L0B2,
          "active6": ACTIVE6, "linac17": LINAC17, "all31": LINAC17 + L0B2 + ACTIVE6 + PASSIVE6}


@torch.no_grad()
def stack(m):
    B = m.base.shape[0]
    imgs, offs = [], []
    for lo in range(0, B, 6):
        a, _, c = m.forward_full(rows=list(range(lo, min(lo + 6, B))))
        imgs.append(a); offs.append(c)
    return torch.cat(imgs, 1), torch.cat(offs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--h", type=float, default=0.02)
    a = ap.parse_args()
    dev = "cuda"
    d = json.loads((OUT / f"{a.arm}.json").read_text())
    r = best_restart(d)
    m, cfg, _ = make_model(d["arm"], dev, seed=r["seed"])
    m.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in r["state"].items()},
                      strict=False)
    real = orient_targets(load_real(dev, ARMS[d["arm"]]["targets"]), True, True)
    w = real["weights"].double()                                          # (2, B)
    w_off = float(cfg.get("w_off", 0.1)) * real["off_weights"].double()   # (B,)
    sw = torch.cat([w.reshape(-1).repeat_interleave(160 * 300),
                    w_off.repeat_interleave(160 * 300)]).sqrt()

    def vec(imgs, offs):
        return torch.cat([imgs.reshape(-1), offs.reshape(-1)]) * sw

    i0, o0 = stack(m)
    res = vec(i0 - real["images"], o0 - real["off"])                      # weighted residual
    # u-space FD: perturb the knob's normalized value by ±h via the raw parameter
    cols = {}
    for i, k in enumerate(m.s.fit_knobs):
        raw0 = float(m.delta_raw[i])
        t0 = np.tanh(raw0)
        b = float(m.s.bound_hi[i] if t0 >= 0 else m.s.bound_lo[i])
        out = []
        for sgn in (+1, -1):
            u_t = float(np.clip(t0 * b + sgn * a.h, -0.999 * float(m.s.bound_lo[i]),
                                0.999 * float(m.s.bound_hi[i])))
            bb = float(m.s.bound_hi[i] if u_t >= 0 else m.s.bound_lo[i])
            m.delta_raw.data[i] = float(np.arctanh(u_t / bb))
            out.append((u_t, stack(m)))
        m.delta_raw.data[i] = raw0
        (up, (ip, op)), (um, (im, om)) = out
        cols[k] = (vec(ip, op) - vec(im, om)) / (up - um)                 # d(weighted img)/du
        print(f"  J[{k}] |col| {float(cols[k].norm()):.3g}", flush=True)
    report = {"arm": a.arm, "h_u": a.h, "residual_norm2": float(res.norm() ** 2),
              "tv_weighted_now": float(r["final_img_loss"]), "groups": {}}
    for g, names in GROUPS.items():
        names = [n for n in names if n in cols]
        if not names:
            continue
        J = torch.stack([cols[n] for n in names], 1)                      # (P, k)
        coef = torch.linalg.lstsq(J, -res.unsqueeze(1)).solution[:, 0]   # du minimizing |res + J du|
        expl = 1.0 - float((res + J @ coef).norm() ** 2 / res.norm() ** 2)
        report["groups"][g] = {"explained_frac_L2": expl,
                               "best_du": dict(zip(names, map(float, coef))),
                               "cos_each": {n: float(-(cols[n] @ res) / (cols[n].norm() * res.norm()))
                                            for n in names}}
        print(f"{g:14s} explained {expl:.3f}  best du "
              + " ".join(f"{n.split(':')[-2] if n.count(':') > 1 else n}={c:+.2f}"
                         for n, c in zip(names, coef)), flush=True)
    (OUT.parent / f"jacobian_alignment_{a.arm}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
