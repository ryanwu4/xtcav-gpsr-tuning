"""Ablation (Ryan's request): recovery WITHOUT the MAP prior (λ=0).

Re-runs the self-recovery scenarios with the SAME targets/seeds as the λ* run
(seeds 200+s, identical make_truth), and the four Bmad-truth scenarios (100k
target replica only), all at λ=0. Everything else identical — n, bandwidth,
iters, optimizer. Comparison quantifies what ridge buys: null-space containment
(per-knob spurious pulls) vs identifiable-direction recovery.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=... python scripts/run_stageC_noridge.py
Outputs -> results/stageC/noridge.json
"""

import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from run_stageC_selfrecovery import SCENARIOS, base_knobs, make_truth, one_fit  # noqa: E402
from run_stageC_bmadrecovery import scan_targets  # noqa: E402
from xtcav_gpsr_tuning.fit import ForwardModel, load_setup, run_fit  # noqa: E402

SCANS = paths.TWOBUNCH_ROOT / "data" / "xtcav23_delta_scans"
OUT = REPO / "results" / "stageC"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_setup(device=device)
    n_p = int(cfg["n_particles"]) // 2
    base = base_knobs(setup).to(device)
    iters = 200

    results = {"lambda": 0.0, "n_particles": n_p, "recovery": [], "bmad": {}}

    # self-recovery, same targets as the lambda* run (seeds 200+s)
    for name, pulls in SCENARIOS.items():
        for seed in (0, 1, 2):
            targets = make_truth(setup, cfg, base, pulls, seed=200 + seed,
                                 n_particles=n_p)
            r = one_fit(setup, cfg, base, targets, lam=0.0, seed=seed,
                        n_particles=n_p, iters=iters)
            results["recovery"].append({
                "scenario": name, "seed": seed, "truth_pulls": pulls,
                "pulls": r["pulls"], "nu_y": r["nu_y"], "dtr": r["dtr"],
                "veff": r["veff"], "img_loss": r["final_img_loss"],
            })
            mx = max(abs(v) for v in r["pulls"].values())
            print(f"  self {name} seed={seed}: img {r['final_img_loss']:.4f} "
                  f"max|pull| {mx:.1f}", flush=True)

    # Bmad-truth, 100k targets only
    for scen_dir in sorted(p for p in SCANS.iterdir()
                           if p.is_dir() and p.name != "scratch"):
        scen = scen_dir.name
        man = json.loads((scen_dir / "manifest.json").read_text())
        n_steps = len(man["steps"])
        grid_norm = man["l2_norm_grid"][:n_steps]
        b = torch.full((n_steps, 23), 0.5, dtype=torch.float64)
        b[:, setup.norm["knob_keys"].index("L2PhaseSet")] = torch.tensor(grid_norm)
        b = b.to(device)
        t_imgs, t_splits = scan_targets(scen_dir, setup.grid, n_steps)
        model = ForwardModel(setup, b, float(cfg["voltage_V"]), n_particles=n_p,
                             fit_veff=(scen == "veff"), seed=0).to(device)
        r = run_fit(model, t_imgs.to(device), t_splits.to(device), lam=0.0,
                    iters=iters, lr=0.04, w_q=cfg.get("w_charge_split", 0.1))
        results["bmad"][scen] = {"delta_true_phys": man["delta_phys"],
                                 "voltage_true_V": man["voltage_V"], "fit": r}
        mx = max(abs(v) for v in r["pulls"].values())
        print(f"  bmad {scen}: max|pull| {mx:.1f} nu_y {r['nu_y']:.3f} "
              f"veff {r['veff']/1e6:.2f} MV", flush=True)

    (OUT / "noridge.json").write_text(json.dumps(results, indent=1))
    print(f"done -> {OUT / 'noridge.json'}")


if __name__ == "__main__":
    main()
