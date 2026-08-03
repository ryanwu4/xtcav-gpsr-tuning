"""Stage-C C3: flow→flow self-recovery, λ L-curve, identifiability SVD.

Targets ALWAYS use an independent z-draw (critic finding 1); one same-z run is
included as the optimizer sanity check only. Runs on GPU if available.

  PYTHONPATH=$PWD/src:$TWOBUNCH/src CUDA_VISIBLE_DEVICES=0 \
    python scripts/run_stageC_selfrecovery.py [--quick]

Outputs -> results/stageC/selfrecovery/{lcurve.json, recovery.json,
identifiability.json, gradient_noise.json}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from xtcav_gpsr_tuning.fit import ForwardModel, load_setup, map_loss, run_fit  # noqa: E402
from xtcav_gpsr_tuning.identifiability import jacobian_svd, save  # noqa: E402

OUT = REPO / "results" / "stageC" / "selfrecovery"

SCENARIOS = {
    "mover": {"S1EL_xOffset": 1.5, "S2ER_xOffset": -2.0, "S2EL_yOffset": 1.0},
    "energy": {"L2EnergyOffset": 1.5, "L3EnergyOffset": -1.5},
    "quad": {"Q5FFkG": 1.5, "Q3FFkG": -1.0},
    "mixed": {"S2ER_xOffset": 1.5, "S1EL_yOffset": 1.0, "L3EnergyOffset": 1.0,
              "Q4FFkG": -1.0},
}


def base_knobs(setup, n_steps=7):
    base = torch.full((n_steps, 23), 0.5, dtype=torch.float64)
    j = setup.norm["knob_keys"].index("L2PhaseSet")
    base[:, j] = torch.linspace(0.42, 0.58, n_steps)  # ~±1 deg L2 mini-scan
    return base


def make_truth(setup, cfg, base, pulls: dict, seed, n_particles, anchor="com"):
    truth = ForwardModel(setup, base, cfg["voltage_V"], n_particles=n_particles,
                         anchor=anchor, seed=seed).to(base.device)
    with torch.no_grad():
        for k, p in pulls.items():
            j = truth.s.fit_knobs.index(k)
            val = p * float(truth.s.sigma_norm[j]) / float(truth.s.bound_norm[j])
            truth.delta_raw[j] = torch.atanh(torch.tensor(val).clamp(-0.99, 0.99))
        imgs, splits = truth()
    return imgs.detach(), splits.detach()


def one_fit(setup, cfg, base, targets, lam, seed, n_particles, anchor="com",
            iters=150, flip_targets=False):
    t_imgs, t_splits = targets
    if flip_targets:
        t_imgs = torch.flip(t_imgs, dims=[-1])  # streak-sign/orientation hypothesis test
    model = ForwardModel(setup, base, cfg["voltage_V"], n_particles=n_particles,
                         anchor=anchor, seed=seed).to(base.device)
    res = run_fit(model, t_imgs, t_splits, lam=lam, iters=iters, lr=0.04,
                  w_q=cfg.get("w_charge_split", 0.1))
    with torch.no_grad():
        parts = map_loss(model, t_imgs, t_splits, lam)
    res["final_img_loss"] = float(parts["img"])
    res["final_prior"] = float(parts["prior"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced grid for smoke")
    ap.add_argument("--n-particles", type=int, default=None)
    ap.add_argument("--iters", type=int, default=150)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_setup(device=device)
    n_p = args.n_particles or (4096 if args.quick else int(cfg["n_particles"]) // 2)
    OUT.mkdir(parents=True, exist_ok=True)
    base = base_knobs(setup).to(device)
    lam_grid = cfg["lambda_grid"][:2] if args.quick else cfg["lambda_grid"]
    seeds = [0, 1] if args.quick else [0, 1, 2]

    # ---------------- identifiability SVD (pre-registered cut) --------------- #
    print("== identifiability SVD ==", flush=True)
    m0 = ForwardModel(setup, base, cfg["voltage_V"], n_particles=2048, seed=0).to(device)
    ident = jacobian_svd(m0, cut=0.05)
    save(ident, OUT / "identifiability.json")
    lev = ident["leverage"]
    weak = [k for k, v in lev.items() if v < 0.05]
    print(f"  kept {ident['n_kept']}/{len(ident['param_names'])} directions; "
          f"low-leverage params: {weak}", flush=True)

    # ---------------- same-z sanity (optimizer must reach ~0) ---------------- #
    print("== same-z sanity ==", flush=True)
    truth_sane = make_truth(setup, cfg, base, SCENARIOS["mixed"], seed=0,
                            n_particles=n_p)
    sane = one_fit(setup, cfg, base, truth_sane, lam=0.0, seed=0,
                   n_particles=n_p, iters=args.iters)
    print(f"  same-z final img loss {sane['final_img_loss']:.4f} (target ~0)", flush=True)
    (OUT / "same_z_sanity.json").write_text(json.dumps(sane, indent=1))

    # ---------------- λ L-curve, both conventions ---------------------------- #
    results = {"lcurve": [], "recovery": [], "flipped": None, "jitter_blur": None,
               "n_particles": n_p, "config_bandwidth_um": cfg["bandwidth_um"]}
    for anchor in ("com", "major_lobe"):
        for lam in lam_grid:
            for seed in seeds[:2]:  # L-curve on 2 seeds; per-scenario recovery keeps all
                targets = make_truth(setup, cfg, base, SCENARIOS["mixed"],
                                     seed=100 + seed, n_particles=n_p, anchor=anchor)
                r = one_fit(setup, cfg, base, targets, lam, seed, n_p,
                            anchor=anchor, iters=args.iters)
                truth_pulls = SCENARIOS["mixed"]
                err = {k: r["pulls"][k] - truth_pulls.get(k, 0.0)
                       for k in r["pulls"]}
                results["lcurve"].append({
                    "anchor": anchor, "lambda": lam, "seed": seed,
                    "img_loss": r["final_img_loss"], "prior": r["final_prior"],
                    "pull_err_rms": float(np.sqrt(np.mean(np.array(list(err.values())) ** 2))),
                    "pulls": r["pulls"], "nu_y": r["nu_y"], "dtr": r["dtr"],
                })
                print(f"  lcurve {anchor} lam={lam:g} seed={seed}: "
                      f"img {r['final_img_loss']:.4f} pull_err_rms "
                      f"{results['lcurve'][-1]['pull_err_rms']:.2f}", flush=True)

    # choose λ: smallest pull_err_rms averaged over seeds, com convention
    import collections
    agg = collections.defaultdict(list)
    for r in results["lcurve"]:
        if r["anchor"] == "com":
            agg[r["lambda"]].append(r["pull_err_rms"])
    lam_star = min(agg, key=lambda L: np.mean(agg[L]))
    results["lambda_star"] = lam_star
    print(f"== chosen lambda* = {lam_star:g} ==", flush=True)

    # ---------------- per-scenario recovery at λ* ---------------------------- #
    for name, pulls in SCENARIOS.items():
        for seed in seeds:
            targets = make_truth(setup, cfg, base, pulls, seed=200 + seed,
                                 n_particles=n_p)
            r = one_fit(setup, cfg, base, targets, lam_star, seed, n_p,
                        iters=args.iters)
            results["recovery"].append({
                "scenario": name, "seed": seed, "truth_pulls": pulls,
                "pulls": r["pulls"], "nu_y": r["nu_y"], "dtr": r["dtr"],
                "veff": r["veff"], "img_loss": r["final_img_loss"],
            })
            print(f"  recover {name} seed={seed}: img {r['final_img_loss']:.4f}",
                  flush=True)

    # ---------------- flipped-target discrete-hypothesis test ---------------- #
    targets = make_truth(setup, cfg, base, SCENARIOS["mixed"], seed=300,
                         n_particles=n_p)
    good = one_fit(setup, cfg, base, targets, lam_star, 0, n_p, iters=args.iters)
    flipped = one_fit(setup, cfg, base, targets, lam_star, 0, n_p,
                      iters=args.iters, flip_targets=True)
    results["flipped"] = {"correct_img_loss": good["final_img_loss"],
                          "flipped_img_loss": flipped["final_img_loss"],
                          "separation_ok": flipped["final_img_loss"]
                          > 1.5 * good["final_img_loss"]}
    print(f"  flipped-target: correct {good['final_img_loss']:.4f} vs flipped "
          f"{flipped['final_img_loss']:.4f}", flush=True)

    # ---------------- jitter-blurred target ---------------------------------- #
    # target = mean image over a ±0.6deg-jittered phase ensemble (Stage-D per-step
    # mean rehearsal); fit with the sharp model — bounds the blur-induced bias.
    imgs_acc, splits_acc = None, None
    rng = np.random.default_rng(7)
    for _ in range(8):
        phi = float(rng.normal(0.0, 0.6))
        t = ForwardModel(setup, base, cfg["voltage_V"], n_particles=n_p // 2,
                         phi_deg=phi, seed=int(rng.integers(1 << 30))).to(device)
        with torch.no_grad():
            im, sp = t()
        imgs_acc = im if imgs_acc is None else imgs_acc + im
        splits_acc = sp if splits_acc is None else splits_acc + sp
    targets = (imgs_acc / 8.0, splits_acc / 8.0)
    r = one_fit(setup, cfg, base, targets, lam_star, 0, n_p, iters=args.iters)
    results["jitter_blur"] = {"pulls": r["pulls"], "nu_y": r["nu_y"],
                              "veff": r["veff"], "dtr": r["dtr"],
                              "img_loss": r["final_img_loss"],
                              "note": "truth deltas = 0; nonzero pulls = blur bias"}
    print(f"  jitter-blur: img {r['final_img_loss']:.4f} max|pull| "
          f"{max(abs(v) for v in r['pulls'].values()):.2f}", flush=True)

    (OUT / "selfrecovery.json").write_text(json.dumps(results, indent=1))
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
