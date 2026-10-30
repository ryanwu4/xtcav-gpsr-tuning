"""Stage-D2 light gate: sim-to-sim self-recovery smoke test for the reachability fit.

Targets = the C_B3 forward model itself at a KNOWN injected state, rendered with an
INDEPENDENT particle seed (the fit's own z-buffers are never used for targets), on
the real 15673 scan geometry and loss weights. The A1 fit (31 knobs, latent at the
prior mean, λ_reach) then starts from δ=0. Checks: the loss falls to near the
independent-noise floor (truth model vs targets), and the injected directions move
the right way. Optional --latent-truth draws the target latent away from the prior
mean (shared, 1σ) to see what an A1 fit (latent fixed at the mean) does with it.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src python scripts/run_stageD2_selfrecovery.py
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
from run_stageD2_reach import OUT, evaluate, fit_once, get_setup, make_model  # noqa: E402
from run_stageD_fit import orient_targets  # noqa: E402
from xtcav_gpsr_tuning.stage_d import load_targets  # noqa: E402

# injected truth: DELTAS from the baseline, physical units (a mix of linac, mover and
# injector directions; drive_fraction 0.75 -> 0.71, MTE 400 -> 430 meV)
TRUTH = {"S2EL_xOffset": 300e-6, "S1EL_yOffset": -300e-6, "L2EnergyOffset": 8e6,
         "laser_timing_ps": 0.15, "GUNF:theta0_deg": 0.3, "drive_fraction": -0.04,
         "distgen:start:MTE:value": 30.0}


def truth_model(device, latent_truth):
    m, cfg, _ = make_model("A2_D" if latent_truth else "A1_D", device, seed=97)
    keys = m.s.norm["knob_keys"]
    lo, hi = np.array(m.s.norm["knob_low"]), np.array(m.s.norm["knob_high"])
    with torch.no_grad():
        for name, v in TRUTH.items():
            j = keys.index(name)
            i = m.s.fit_knobs.index(name)
            u = v / (hi[j] - lo[j])
            b = float((m.s.bound_hi if u >= 0 else m.s.bound_lo)[i])
            m.delta_raw[i] = torch.atanh(torch.tensor(u / b))
        if latent_truth:  # one prior σ away from the scan-averaged prior mean, per dim
            g = torch.Generator().manual_seed(5)        # (s_lat is in prior-σ units)
            m.s_lat.add_(torch.randn(m.s_lat.shape, generator=g, dtype=torch.float64).to(m.s_lat))
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--latent-truth", action="store_true")
    args = ap.parse_args()
    device = "cuda"
    cfg, _, _ = get_setup("cnf", device)
    real = orient_targets(load_targets(cfg, device=device), True, True)
    tm, cfg = truth_model(device, args.latent_truth)
    assert torch.isfinite(tm.delta_raw).all(), "injected truth outside the δ bound"
    with torch.no_grad():
        B = tm.base.shape[0]
        imgs, splits, offs = [], [], []
        for lo in range(0, B, 6):
            rows = list(range(lo, min(lo + 6, B)))
            a, b, c = tm.forward_full(rows=rows)
            imgs.append(a); splits.append(b); offs.append(c)
        tgt = {"images": torch.cat(imgs, 1), "splits": torch.cat(splits, 1),
               "off": torch.cat(offs, 0), "mask": real["mask"], "weights": real["weights"],
               "off_weights": real["off_weights"]}
    floor = evaluate(make_truth_twin(tm, device, args.latent_truth), tgt, cfg)
    r = fit_once("A1_D", tgt, device, seed=0, iters=args.iters)
    span = {k: float(np.array(tm.s.norm["knob_high"])[tm.s.norm["knob_keys"].index(k)]
                     - np.array(tm.s.norm["knob_low"])[tm.s.norm["knob_keys"].index(k)])
            for k in TRUTH}
    recov = {}
    for k, truth_d in TRUTH.items():
        recov[k] = {"truth_delta": truth_d, "fit_delta": r["delta_phys"][k],
                    "ratio": r["delta_phys"][k] / truth_d,
                    "truth_pull": truth_d / float(
                        tm.s.sigma_norm[tm.s.fit_knobs.index(k)] * span[k])}
    out = {"truth": TRUTH, "latent_truth": args.latent_truth,
           "floor_truth_model_vs_targets": floor["img"], "null_img": r["null"]["img"],
           "fit_img": r["final_img_loss"], "recovery": recov,
           "walls": r["walls"], "top_pulls": dict(sorted(r["pulls"].items(),
                                                         key=lambda kv: -abs(kv[1]))[:10]),
           "nu_y": r["nu_y"], "dtr": r["dtr"], "l1_shift_deg": r["l1_shift_deg"],
           "l2_shift_deg": r["l2_shift_deg"]}
    name = "selfrecovery_latent.json" if args.latent_truth else "selfrecovery.json"
    (OUT.parent / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def make_truth_twin(tm, device, latent_truth):
    """The truth state at the FIT's particle seed (seed 0): its loss against the
    independent-seed targets is the floor a perfect fit can reach."""
    m, _, _ = make_model("A2_D" if latent_truth else "A1_D", device, seed=0)
    m.load_state_dict({k: v for k, v in tm.state_dict().items() if k not in ("zd", "zw")},
                      strict=False)
    return m


if __name__ == "__main__":
    main()
