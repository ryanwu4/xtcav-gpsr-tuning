"""Stage-C C4: fit the MAP harness against the Bmad δ-injected mini-scans.

Consumes twobunch-s2e-rl/data/xtcav23_delta_scans/<scenario>/ (make_delta_scans.py).
The fit sees the UNPERTURBED base knobs (box center + the L2 grid) — exactly the
"recorded settings" role — and must recover δ_true through the flow+segment
forward model. Per scenario: fits against the 100k target AND two independent
10k-subsample replicas (error bars); veff scenario runs with fit_veff=True.

Bias is reported per knob; the REPORT step projects onto the pre-registered
identifiable subspace (identifiability.json) and references the statistics-
matched C3 baseline — do not interpret raw per-knob bias on null-space dims.

  PYTHONPATH=$PWD/src:$TWOBUNCH/src CUDA_VISIBLE_DEVICES=0 \
    python scripts/run_stageC_bmadrecovery.py

Outputs -> results/stageC/bmadrecovery.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from xtcav_gpsr_tuning.fit import ForwardModel, load_setup, run_fit  # noqa: E402
from xtcav_gpsr_tuning.screen import ScreenGrid  # noqa: E402
from xtcav_gpsr_tuning.targets import load_dtotr_target, target_charge_split  # noqa: E402

SCANS = paths.TWOBUNCH_ROOT / "data" / "xtcav23_delta_scans"
OUT = REPO / "results" / "stageC"
SELF = OUT / "selfrecovery" / "selfrecovery.json"


def scan_targets(scen_dir: Path, grid: ScreenGrid, n_steps: int,
                 subsample=None, seed=0):
    imgs, splits = [], []
    for sign_case in ("plus", "minus"):
        sign_imgs = []
        for k in range(n_steps):
            h5 = scen_dir / f"step{k}_DTOTR_{sign_case}.h5"
            sign_imgs.append(load_dtotr_target(h5, grid, subsample=subsample,
                                               seed=seed + 17 * k))
        imgs.append(torch.stack(sign_imgs))
    split = np.mean([target_charge_split(scen_dir / f"step{k}_DTOTR_plus.h5")
                     for k in range(n_steps)])
    t_imgs = torch.stack(imgs)  # (2, n_steps, ny, nx)
    t_splits = torch.full((2, n_steps), float(split), dtype=torch.float64)
    return t_imgs, t_splits


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of scenarios to (re)fit")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_setup(device=device)
    lam_star = json.loads(SELF.read_text())["lambda_star"] if SELF.exists() else 1e-3
    n_p = int(cfg["n_particles"]) // 2

    out_path = OUT / "bmadrecovery.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else \
        {"lambda": lam_star, "n_particles": n_p, "scenarios": {}}
    for scen_dir in sorted(p for p in SCANS.iterdir() if p.is_dir() and p.name != "scratch"):
        scen = scen_dir.name
        if only and scen not in only:
            continue
        man = json.loads((scen_dir / "manifest.json").read_text())
        n_steps = len(man["steps"])
        if n_steps == 0:
            print(f"[{scen}] no completed steps yet — skipping")
            continue
        grid_norm = man["l2_norm_grid"][:n_steps]
        base = torch.full((n_steps, 23), 0.5, dtype=torch.float64)
        base[:, setup.norm["knob_keys"].index("L2PhaseSet")] = torch.tensor(grid_norm)
        base = base.to(device)
        fit_veff = scen == "veff"
        v_init = float(cfg["voltage_V"])  # init at nominal 22 MV even for veff scenario

        fits = []
        for rep, sub in ((0, None), (1, 10000), (2, 10000)):
            t_imgs, t_splits = scan_targets(scen_dir, setup.grid, n_steps,
                                            subsample=sub, seed=1000 * rep)
            model = ForwardModel(setup, base, v_init, n_particles=n_p,
                                 fit_veff=fit_veff, seed=rep).to(device)
            r = run_fit(model, t_imgs.to(device), t_splits.to(device),
                        lam=lam_star, iters=200, lr=0.04,
                        w_q=cfg.get("w_charge_split", 0.1))
            fits.append(r)
            print(f"[{scen}] replica {rep} ({'100k' if sub is None else '10k'}): "
                  f"nu_y {r['nu_y']:.4f} veff {r['veff']/1e6:.2f} MV", flush=True)

        results["scenarios"][scen] = {
            "delta_true_phys": man["delta_phys"],
            "voltage_true_V": man["voltage_V"],
            "fits": fits,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bmadrecovery.json").write_text(json.dumps(results, indent=1))
    print(f"done -> {OUT / 'bmadrecovery.json'}")


if __name__ == "__main__":
    main()
