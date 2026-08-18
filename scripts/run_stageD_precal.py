"""Stage-D D0.2–D0.5 pre-calibration on the REAL 15673 geometry (sim-to-sim).

All runs use the 15673-configured segment, the calibrated V_eff, the v2 frozen
1-px kernel + sdelta/ex physics nuisances (n=65536), image-space COM anchoring,
and the real 33-step × ±sign conditioning — i.e. the exact forward model the
real fit will use. Targets here
are FLOW-GENERATED with known truth (independent z-draws); the real targets
enter only via their split-half noise floors (discrepancy cross-check).

Stages (each writes its own JSON; --stage to run one, default all):
  register — D0.3 Jacobian SVD on the 66-image geometry (pre-registered subspace)
             + per-param TV-unit signal table (S/N vs the D1 split-half floors)
  lambda   — D0.4 λ L-curve (image_com; anchor robustness established in v1)
  control  — D0.2 high-stat-target control (smooth-target systematic on kept dirs)
  nuy      — D0.5 ν_y jitter replication (5 seeds, ±0.6° ensemble targets)

  cd /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning && \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src:/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/run_stageD_precal.py [--stage X] [--quick]

Outputs -> results/stageD/15673/precal/{register,lambda,control,nuy}.json
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
from xtcav_gpsr_tuning.fit import (ForwardModel, eval_loss_chunked,  # noqa: E402
                                   run_fit_chunked)
from xtcav_gpsr_tuning.identifiability import _images_at, jacobian_svd, save  # noqa: E402
from xtcav_gpsr_tuning.stage_d import (conditioning, load_stage_d,  # noqa: E402
                                       load_targets, provenance)

OUT = REPO / "results" / "stageD" / "15673" / "precal"

# 1–2σ_j injection scenarios (Stage-C set, reused for geometric comparability).
# v2 adds an sdelta-recovery scenario ("__sdelta__" is the truth's log_sdelta,
# not a knob) — the new headline nuisance must be shown recoverable sim-to-sim.
SCENARIOS = {
    "zero": {},
    "mixed": {"S2ER_xOffset": 1.5, "S1EL_yOffset": 1.0, "L3EnergyOffset": 1.0,
              "Q4FFkG": -1.0},
    "sdelta": {"__sdelta__": 0.69},  # truth sdelta = 2.0x (the observed scale)
}
CHUNK = 3


def make_model(setup, cfg, base, n_particles, seed, anchor_mode="image_com",
               phi_deg=0.0):
    """v2: image_com only (targets exist only in that convention; anchor
    robustness of λ* was established in v1/Stage C). sdelta/ex enabled."""
    kw = dict(n_particles=n_particles, seed=seed, phi_deg=phi_deg,
              fit_sdelta=True, fit_ex=True, l2ref_nuisance=True)
    if anchor_mode == "image_com":
        kw.update(anchor="none", image_anchor=True)
    else:
        kw.update(anchor=anchor_mode)
    return ForwardModel(setup, base, cfg["voltage_V"], **kw).to(base.device)


def make_truth(setup, cfg, base, pulls, seed, n_particles,
               anchor_mode="image_com", phi_deg=0.0):
    truth = make_model(setup, cfg, base, n_particles, seed, anchor_mode, phi_deg)
    with torch.no_grad():
        for k, p in pulls.items():
            if k == "__sdelta__":
                truth.log_sdelta.copy_(torch.tensor(float(p)))
                continue
            j = truth.s.fit_knobs.index(k)
            val = p * float(truth.s.sigma_norm[j]) / float(truth.s.bound_norm[j])
            truth.delta_raw[j] = torch.atanh(torch.tensor(val).clamp(-0.99, 0.99))
        B = base.shape[0]
        imgs, splits = [], []
        for lo in range(0, B, 2 * CHUNK):  # chunked no-grad render (64k parts)
            rows = list(range(lo, min(lo + 2 * CHUNK, B)))
            im, sp = truth(rows=rows)
            imgs.append(im); splits.append(sp)
    return (torch.cat(imgs, dim=1).detach(), torch.cat(splits, dim=1).detach())


def one_fit(setup, cfg, base, targets, lam, seed, n_particles,
            anchor_mode="image_com", iters=150):
    t_imgs, t_splits = targets
    model = make_model(setup, cfg, base, n_particles, seed, anchor_mode)
    res = run_fit_chunked(
        model, t_imgs, t_splits, lam=lam, iters=iters, lr=0.04,
        w_q=cfg.get("w_charge_split", 0.1), chunk=CHUNK,
        nu_y_sigma_ln=cfg.get("nu_y_prior_sigma"),
        sdelta_sigma_ln=cfg.get("sdelta_prior_sigma_ln"),
        ex_sigma_ln=cfg.get("ex_prior_sigma_ln"),
        dtr_sigma_frac=cfg.get("dtr_prior_sigma_frac"),
        l2ref_sigma_deg=float(cfg["l2ref_sigma_deg"]))
    parts = eval_loss_chunked(model, t_imgs, t_splits, chunk=2 * CHUNK)
    res["final_img_loss"] = parts["img"]
    res["final_prior"] = float(((model.delta_norm / model.s.sigma_norm) ** 2
                                ).sum().detach())
    res["l2_shift_deg"] = float(model.l2_shift_deg)
    return res


def stage_register(setup, cfg, base, real, quick):
    print("== D0.3 register: Jacobian SVD on the 66-image geometry ==", flush=True)
    m0 = make_model(setup, cfg, base, 2048 if quick else 4096, seed=0)
    ident = jacobian_svd(m0, cut=0.05)

    # per-param TV-unit signal (mean per-image TV of a ±1-unit prior-scaled
    # perturbation) — converts the D0.2 systematic into TV units (critic
    # minor-9) and re-measures S/N against the REAL split-half floors
    n_par = len(ident["param_names"])
    base_imgs = _images_at(m0, np.zeros(n_par))
    tv_signal = {}
    for j, name in enumerate(ident["param_names"]):
        tp = np.zeros(n_par); tp[j] = 1.0
        d = (_images_at(m0, tp) - base_imgs).abs().sum(dim=(-2, -1))
        tv_signal[name] = float(d.mean())
    _images_at(m0, np.zeros(n_par))
    floor_med = float(torch.nanmedian(real["floors"]))
    ident["tv_signal_per_unit"] = tv_signal
    ident["real_floor_tv_median"] = floor_med
    ident["snr_vs_floor"] = {k: v / floor_med for k, v in tv_signal.items()}
    ident["provenance"] = provenance()
    save(ident, OUT / "register.json")
    lev = ident["leverage"]
    print(f"  kept {ident['n_kept']}/{n_par}; low-leverage (<0.05): "
          f"{[k for k, v in lev.items() if v < 0.05]}", flush=True)
    print(f"  TV S/N vs floor (median floor {floor_med:.3f}): "
          f"{ {k: round(v, 2) for k, v in ident['snr_vs_floor'].items()} }",
          flush=True)


def stage_lambda(setup, cfg, base, real, quick, n_p, iters, seeds):
    print("== D0.4 lambda re-calibration ==", flush=True)
    results = {"lcurve": [], "n_particles": n_p,
               "real_floor_tv_median": float(torch.nanmedian(real["floors"]))}
    grid = cfg["lambda_grid"][:2] if quick else cfg["lambda_grid"]
    for anchor_mode in ("image_com",):
        for lam in grid:
            for seed in seeds[:2]:
                targets = make_truth(setup, cfg, base, SCENARIOS["mixed"],
                                     seed=100 + seed, n_particles=n_p,
                                     anchor_mode=anchor_mode)
                r = one_fit(setup, cfg, base, targets, lam, seed, n_p,
                            anchor_mode=anchor_mode, iters=iters)
                err = {k: r["pulls"][k] - SCENARIOS["mixed"].get(k, 0.0)
                       for k in r["pulls"]}
                rec = {"anchor": anchor_mode, "lambda": lam, "seed": seed,
                       "img_loss": r["final_img_loss"], "prior": r["final_prior"],
                       "pull_err_rms": float(np.sqrt(np.mean(
                           np.array(list(err.values())) ** 2))),
                       "pulls": r["pulls"], "nu_y": r["nu_y"], "dtr": r["dtr"],
                       "sdelta": r["sdelta"], "ex": r["ex"]}
                results["lcurve"].append(rec)
                print(f"  {anchor_mode} lam={lam:g} seed={seed}: img "
                      f"{rec['img_loss']:.4f} pull_err {rec['pull_err_rms']:.2f}",
                      flush=True)
    import collections
    agg = collections.defaultdict(list)
    for r in results["lcurve"]:
        if r["anchor"] == "image_com":
            agg[r["lambda"]].append(r["pull_err_rms"])
    results["lambda_star"] = min(agg, key=lambda L: np.mean(agg[L]))
    # discrepancy principle: fitted residual should approach but not undercut
    # the real split-half floor (fitting real data below the floor = noise fit)
    results["discrepancy_note"] = (
        "compare lcurve img_loss at lambda_star to real_floor_tv_median; "
        "the REAL fit's residual should land near/above the floor")
    results["provenance"] = provenance()
    (OUT / "lambda.json").write_text(json.dumps(results, indent=1))
    print(f"== lambda* = {results['lambda_star']:g} ==", flush=True)


def stage_control(setup, cfg, base, quick, n_p, iters, seeds, lam_star):
    print("== D0.2 high-stat-target control ==", flush=True)
    n_target = n_p * (2 if quick else 8)  # LOW-noise targets
    results = {"lambda": lam_star, "n_fit": n_p, "n_target": n_target,
               "runs": []}
    for name, pulls in SCENARIOS.items():
        for seed in seeds[:2]:
            targets = make_truth(setup, cfg, base, pulls, seed=400 + seed,
                                 n_particles=n_target)
            r = one_fit(setup, cfg, base, targets, lam_star, seed, n_p,
                        iters=iters)
            err = {k: r["pulls"][k] - pulls.get(k, 0.0) for k in r["pulls"]}
            results["runs"].append({
                "scenario": name, "seed": seed, "truth_pulls": pulls,
                "pulls": r["pulls"], "pull_err": err, "nu_y": r["nu_y"],
                "sdelta": r["sdelta"], "ex": r["ex"],
                "sdelta_err_ln": float(np.log(r["sdelta"]))
                                 - pulls.get("__sdelta__", 0.0),
                "dtr": r["dtr"], "img_loss": r["final_img_loss"]})
            print(f"  {name} seed={seed}: img {r['final_img_loss']:.4f} "
                  f"max|pull_err| {max(abs(v) for v in err.values()):.2f} "
                  f"sdelta {r['sdelta']:.2f} ex {r['ex']:.2f}", flush=True)
    # controlled systematic = rms pull error over runs, per knob
    knobs = list(results["runs"][0]["pull_err"])
    sys_rms = {k: float(np.sqrt(np.mean([r["pull_err"][k] ** 2
                                          for r in results["runs"]])))
               for k in knobs}
    results["systematic_rms_pulls"] = sys_rms
    results["gate_note"] = ("escalate to Stage-E flow fine-tune if kept-direction "
                            "systematic exceeds ~0.5 sigma (evaluate on the "
                            "register.json subspace)")
    results["provenance"] = provenance()
    (OUT / "control.json").write_text(json.dumps(results, indent=1))
    print(f"  max systematic rms pull: {max(sys_rms.values()):.2f}", flush=True)


def stage_nuy(setup, cfg, base, quick, n_p, iters, lam_star):
    print("== D0.5 nu_y jitter replication ==", flush=True)
    results = {"lambda": lam_star, "runs": []}
    n_seeds = 2 if quick else 3
    rng = np.random.default_rng(7)
    for s in range(n_seeds):
        # target = mean over a ±0.6°-jittered 8-member phase ensemble, truth δ=0
        imgs_acc, splits_acc = None, None
        for _ in range(8):
            phi = float(rng.normal(0.0, 0.6))
            im, sp = make_truth(setup, cfg, base, {}, seed=int(rng.integers(1 << 30)),
                                n_particles=n_p // 2, phi_deg=phi)
            imgs_acc = im if imgs_acc is None else imgs_acc + im
            splits_acc = sp if splits_acc is None else splits_acc + sp
        r = one_fit(setup, cfg, base, (imgs_acc / 8.0, splits_acc / 8.0),
                    lam_star, s, n_p, iters=iters)
        results["runs"].append({"seed": s, "nu_y": r["nu_y"],
                                "pulls": r["pulls"], "veff": r["veff"],
                                "dtr": r["dtr"], "img_loss": r["final_img_loss"]})
        print(f"  seed {s}: nu_y {r['nu_y']:.4f} max|pull| "
              f"{max(abs(v) for v in r['pulls'].values()):.2f}", flush=True)
    nus = [r["nu_y"] for r in results["runs"]]
    results["nu_y_median"] = float(np.median(nus))
    results["nu_y_spread"] = float(np.ptp(nus))
    results["decision_note"] = (
        "if nu_y bias is consistent in sign and >~2%, enable the 3-point "
        "phi_ensemble in fit_stageD_15673.yaml; else leave null")
    results["provenance"] = provenance()
    (OUT / "nuy.json").write_text(json.dumps(results, indent=1))
    print(f"  nu_y median {results['nu_y_median']:.4f} "
          f"spread {results['nu_y_spread']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["register", "lambda", "control", "nuy"],
                    default=None, help="run one stage (default: all in order)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--iters", type=int, default=150)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_stage_d(device=device)
    print(f"device={device}  V_eff={cfg['voltage_V']/1e6:.3f} MV  "
          f"bandwidth={cfg['bandwidth_um']} um", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    base = conditioning(setup, cfg)
    real = load_targets(cfg, device=device)
    n_p = 8192 if args.quick else int(cfg["n_particles"])
    iters = 60 if args.quick else args.iters
    seeds = [0, 1] if args.quick else [0, 1, 2]

    stages = [args.stage] if args.stage else ["register", "lambda", "control", "nuy"]
    if "register" in stages:
        stage_register(setup, cfg, base, real, args.quick)
    if "lambda" in stages:
        stage_lambda(setup, cfg, base, real, args.quick, n_p, iters, seeds)
    lam_star = cfg["lambda_grid"][2]  # default 3e-3 (Stage-C λ*)
    lam_file = OUT / "lambda.json"
    if lam_file.exists():
        lam_star = float(json.loads(lam_file.read_text())["lambda_star"])
    if "control" in stages:
        stage_control(setup, cfg, base, args.quick, n_p, iters, seeds, lam_star)
    if "nuy" in stages:
        stage_nuy(setup, cfg, base, args.quick, n_p, iters, lam_star)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
