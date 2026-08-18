"""Stage-D D3: the real-data MAP fit on E300_15673 (pre-registered protocol).

GATED: refuses to run unless results/stageD/15673/precal/{register,lambda}.json
exist (D0 discipline — subspace + λ* must be pre-registered before any real fit).

Protocol (plan D3):
  baselines — δ=0 null and (ν_y + L2_ref)-only fit; TV per step recorded
  orient    — full MAP on all 4 orientation/pairing combos (restart 0);
              winner/runner-up ratio ≥ 2 required to fix the hypothesis,
              else both are carried through the ensemble
  ensemble  — ≥8 restarts of the full MAP at λ*; pulls / physical δ̂ /
              projections on the registered subspace with ensemble spread
  sensitivity — pre-registered variants: Student-t prior, ±10% V_eff,
              bandwidth ×{0.7,1.3}, w_off ∈ {0, 0.3}

  cd /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning && \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src:/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/run_stageD_fit.py [--stage X]

Outputs -> results/stageD/15673/fit/{baselines,orient,ensemble,sensitivity}.json
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
from xtcav_gpsr_tuning.stage_d import (check_provenance, conditioning,  # noqa: E402
                                       load_stage_d, load_targets, orient)

PRECAL = REPO / "results" / "stageD" / "15673" / "precal"
OUT = REPO / "results" / "stageD" / "15673" / "fit"

N_PARTICLES = None  # resolved from cfg["n_particles"] (v2: 65536 at the 1-px kernel)
ITERS = 120         # cosine tail past ~100 iters is negligible; 64k makes iters pricey
CONV_TV_MAX = 1.2   # restart-convergence cut (audit #4: was an undocumented
                    # magic 1.0 living only in the figures script)
CHUNK = 3           # combined forward_full at n=64k: chunk 3 ≈ 50 GB peak
                    # (leaves headroom for other users' processes on the GPU)


def orient_targets(real: dict, flip_rows: bool, swap_signs: bool) -> dict:
    """Apply one orientation/pairing hypothesis to every sign-indexed target
    quantity (the z-lobe split is invariant under the row flip; sign-indexed
    tensors swap axis 0 with the pairing)."""
    def swap(t):
        return t.flip(0) if swap_signs else t
    return {
        "images": orient(real["images"], flip_rows, swap_signs),
        "splits": swap(real["splits"]),
        "weights": swap(real["step_weights"]),
        "off": real["off"].flip(-2) if flip_rows else real["off"],
        "mask": real["mask"].flip(-2) if flip_rows else real["mask"],
        "off_weights": real.get("off_step_weights"),
        # audit #3 (secondary): floors are sign-indexed and MUST follow the
        # pairing swap, else the per-step figure plots the wrong sign's floor
        "floors": swap(real["floors"]),
    }


def make_model(setup, cfg, base, seed, nuisance_only=False,
               veff_scale=1.0, rng=None):
    m = ForwardModel(setup, base, cfg["voltage_V"] * veff_scale,
                     n_particles=int(cfg["n_particles"]), anchor="none",
                     image_anchor=True, l2ref_nuisance=True, seed=seed,
                     base_off_norm=base, fit_sdelta=True, fit_ex=True
                     ).to(base.device)
    if rng is not None:  # restart jitter
        with torch.no_grad():
            m.delta_raw.copy_(torch.tensor(
                rng.normal(0.0, 0.3, size=m.delta_raw.shape)))
            m.dtr.copy_(torch.tensor(rng.normal(0.0, 0.005, size=(3,))))
    if nuisance_only:
        m.delta_raw.requires_grad_(False)
        m.dtr.requires_grad_(False)
    return m


def fit_once(setup, cfg, base, tgt, lam, seed, iters=ITERS, prior="ridge",
             nuisance_only=False, veff_scale=1.0, rng=None):
    model = make_model(setup, cfg, base, seed, nuisance_only=nuisance_only,
                       veff_scale=veff_scale, rng=rng)
    w_off = float(cfg.get("w_off", 0.1))
    res = run_fit_chunked(
        model, tgt["images"], tgt["splits"], lam=lam, iters=iters, lr=0.04,
        w_q=float(cfg.get("w_charge_split", 0.1)), chunk=CHUNK, mask=tgt["mask"],
        step_weights=tgt["weights"], targets_off=tgt["off"], w_off=w_off,
        prior=prior, l2ref_sigma_deg=float(cfg["l2ref_sigma_deg"]),
        nu_y_sigma_ln=cfg.get("nu_y_prior_sigma"),
        sdelta_sigma_ln=cfg.get("sdelta_prior_sigma_ln"),
        ex_sigma_ln=cfg.get("ex_prior_sigma_ln"),
        dtr_sigma_frac=cfg.get("dtr_prior_sigma_frac"),
        off_step_weights=tgt.get("off_weights"))
    parts = eval_loss_chunked(model, tgt["images"], tgt["splits"],
                              mask=tgt["mask"], step_weights=tgt["weights"],
                              targets_off=tgt["off"], w_off=w_off, chunk=2 * CHUNK)
    res["final_img_loss"] = parts["img"]
    res["final_off_loss"] = parts["off"]
    res["final_split_loss"] = parts["split"]
    res["l2_shift_deg"] = float(model.l2_shift_deg)
    res["per_image_tv"] = parts["per_image_tv"].tolist()
    res["seed"] = int(seed)                       # audit #9: re-render exactly
    res["total_loss"] = float(res["loss_history"][-1])  # audit #4: the MAP objective
    return res


def theta_from_result(res, cfg, register) -> np.ndarray:
    """MAP result -> prior-scaled parameter vector matching register.json
    (17 pulls, 3 dtr in 1% units, log nu_y, veff offset in MV)."""
    names = register["param_names"]
    th = []
    for n in names:
        if n.startswith("delta:"):
            th.append(res["pulls"][n.split(":", 1)[1]])
        elif n.startswith("dtr:"):
            th.append(res["dtr"][["Q2FF", "Q1FF", "Q0FF"].index(n.split(":")[1])] / 0.01)
        elif n == "nu_y(log)":
            th.append(float(np.log(res["nu_y"])))
        elif n == "veff(MV)":
            th.append((res["veff"] - cfg["voltage_V"]) / 1e6)
        elif n == "sdelta(ln)":
            th.append(float(np.log(res["sdelta"])))
        elif n == "ex(ln)":
            th.append(float(np.log(res["ex"])))
        elif n == "l2_shift(deg)":
            th.append(float(res["l2_shift_deg"]))
        else:
            raise KeyError(f"register parameter {n!r} has no result mapping")
    return np.asarray(th)


def projections(res, cfg, register) -> list:
    th = theta_from_result(res, cfg, register)
    return [float(np.dot(v, th)) for v in np.asarray(register["kept_directions"])]


def stage_baselines(setup, cfg, base, real, lam):
    """Baselines PER ORIENTATION (audit #3). Previously this ran first and
    hard-coded flip0_swap0 while the ensemble used the winning hypothesis —
    and image TV is not invariant under the row flip or the sign swap, so the
    headline "null -> fit" reduction was quoted across mismatched targets
    (measured: null 1.5205 in flip0_swap0 vs 1.3401 in flip1_swap1, which flips
    the pre-registered systematic gate). The null is a cheap forward eval, so
    all four are computed; the nuisance-only fit runs for the carried
    hypotheses only."""
    print("== D3.1 baselines (per orientation) ==", flush=True)
    sel = json.loads((OUT / "orient.json").read_text())
    carried = [sel["winner"]] + ([sel["ranked"][1]] if sel["carry_both"] else [])
    out = {"carried": carried, "null": {}, "nuisance_only": {}}
    model = make_model(setup, cfg, base, seed=0)          # δ = 0
    for key, meta in sel["runs"].items():
        tgt = orient_targets(real, meta["flip_rows"], meta["swap_signs"])
        parts = eval_loss_chunked(model, tgt["images"], tgt["splits"],
                                  mask=tgt["mask"], step_weights=tgt["weights"],
                                  targets_off=tgt["off"],
                                  off_step_weights=tgt.get("off_weights"),
                                  w_off=float(cfg.get("w_off", 0.1)),
                                  chunk=2 * CHUNK)
        out["null"][key] = {"per_image_tv": parts["per_image_tv"].tolist(),
                            "img_loss": parts["img"], "split_loss": parts["split"],
                            "off_loss": parts["off"]}
        print(f"  null {key}: img {parts['img']:.4f}", flush=True)
    for key in carried:
        meta = sel["runs"][key]
        tgt = orient_targets(real, meta["flip_rows"], meta["swap_signs"])
        r = fit_once(setup, cfg, base, tgt, lam, seed=0, nuisance_only=True)
        out["nuisance_only"][key] = r
        print(f"  nuisance-only {key}: img {r['final_img_loss']:.4f} "
              f"nu_y {r['nu_y']:.3f} l2_shift {r['l2_shift_deg']:+.3f} deg",
              flush=True)
    (OUT / "baselines.json").write_text(json.dumps(out, indent=1))
    return out


def stage_orient(setup, cfg, base, real, lam):
    print("== D3.2 orientation selection ==", flush=True)
    runs = {}
    for fr in (False, True):
        for sw in (False, True):
            key = f"flip{int(fr)}_swap{int(sw)}"
            tgt = orient_targets(real, fr, sw)
            r = fit_once(setup, cfg, base, tgt, lam, seed=0)
            runs[key] = {"flip_rows": fr, "swap_signs": sw,
                         "img_loss": r["final_img_loss"],
                         "total_tail": r["loss_history"][-1],
                         "pulls": r["pulls"], "nu_y": r["nu_y"],
                         "sdelta": r["sdelta"], "ex": r["ex"]}
            print(f"  {key}: img {r['final_img_loss']:.4f}", flush=True)
    ranked = sorted(runs, key=lambda k: runs[k]["img_loss"])
    ratio = runs[ranked[1]]["img_loss"] / max(runs[ranked[0]]["img_loss"], 1e-12)
    sel = {"runs": runs, "ranked": ranked, "winner": ranked[0],
           "runner_up_ratio": ratio,
           "carry_both": ratio < float(cfg.get("orientation_selection_ratio", 2.0))}
    (OUT / "orient.json").write_text(json.dumps(sel, indent=1))
    print(f"  winner {ranked[0]} (runner-up ratio {ratio:.2f}; "
          f"carry_both={sel['carry_both']})", flush=True)
    return sel


def stage_ensemble(setup, cfg, base, real, lam, register):
    sel = json.loads((OUT / "orient.json").read_text())
    hyps = [sel["winner"]] + ([sel["ranked"][1]] if sel["carry_both"] else [])
    print(f"== D3.3 ensemble on {hyps} ==", flush=True)
    results = {"lambda": lam, "n_particles": int(cfg["n_particles"]),   # audit #12
               "iters": ITERS, "chunk": CHUNK,
               "bandwidth_um": float(cfg["bandwidth_um"]),
               "voltage_V": float(cfg["voltage_V"]),
               "hypotheses": {}}
    n_restarts = int(cfg.get("n_restarts", 8))
    for key in hyps:
        fr, sw = sel["runs"][key]["flip_rows"], sel["runs"][key]["swap_signs"]
        tgt = orient_targets(real, fr, sw)
        runs = []
        for s in range(n_restarts):
            rng = np.random.default_rng(1000 + s) if s else None  # restart 0 = zero init
            r = fit_once(setup, cfg, base, tgt, lam, seed=s, rng=rng)
            r["projections"] = projections(r, cfg, register)
            runs.append(r)
            print(f"  {key} restart {s}: img {r['final_img_loss']:.4f} "
                  f"max|pull| {max(abs(v) for v in r['pulls'].values()):.2f} "
                  f"sdelta {r['sdelta']:.2f} ex {r['ex']:.2f} nu_y {r['nu_y']:.3f}",
                  flush=True)
        # audit #4: record BOTH criteria (they disagreed by 2.5x in max|pull|)
        # and keep the totals so the choice can be re-derived from the artifact
        conv_ix = [i for i, r in enumerate(runs)
                   if r["final_img_loss"] < CONV_TV_MAX] or list(range(len(runs)))
        best_total = min(conv_ix, key=lambda i: runs[i]["total_loss"])
        best_img = min(conv_ix, key=lambda i: runs[i]["final_img_loss"])
        # audit #6: spread over CONVERGED restarts (ptp over divergent ones
        # inflated it up to 244x); both are recorded
        proj_all = np.array([r["projections"] for r in runs])
        proj_conv = np.array([runs[i]["projections"] for i in conv_ix])
        results["hypotheses"][key] = {
            "runs": [{k: v for k, v in r.items() if k != "loss_history"}
                     for r in runs],
            "converged_indices": conv_ix,
            "conv_tv_max": CONV_TV_MAX,
            "best_restart": best_total,          # by the MAP objective (canonical)
            "best_restart_by_img": best_img,
            "projection_median": np.median(proj_conv, axis=0).tolist(),
            "projection_spread": np.ptp(proj_conv, axis=0).tolist(),
            "projection_spread_all": np.ptp(proj_all, axis=0).tolist(),
        }
    (OUT / "ensemble.json").write_text(json.dumps(results, indent=1))
    return results


def stage_sensitivity(setup, cfg, base, real, lam, hyp=None):
    sel = json.loads((OUT / "orient.json").read_text())
    key = hyp or sel["winner"]
    fr = sel["runs"][key]["flip_rows"]
    sw = sel["runs"][key]["swap_signs"]
    tgt = orient_targets(real, fr, sw)
    print(f"== D3.5 sensitivity variants (on {key}) ==", flush=True)
    out = {}
    variants = {
        "student_t": dict(prior="student_t"),
        "veff_p10": dict(veff_scale=1.10),
        "veff_m10": dict(veff_scale=0.90),
    }
    for name, kw in variants.items():
        r = fit_once(setup, cfg, base, tgt, lam, seed=0, **kw)
        out[name] = {k: v for k, v in r.items() if k != "loss_history"}
        print(f"  {name}: img {r['final_img_loss']:.4f}", flush=True)
    # bandwidth scan: rebuild the grid via a fresh setup per multiplier
    for mult in (0.7, 1.3):
        import copy
        setup2 = copy.copy(setup)
        from xtcav_gpsr_tuning.screen import ScreenGrid
        setup2.grid = ScreenGrid(bandwidth_m=cfg["bandwidth_um"] * mult * 1e-6,
                                 device=setup.device)
        r = fit_once(setup2, cfg, base, tgt, lam, seed=0)
        out[f"bandwidth_x{mult}"] = {k: v for k, v in r.items()
                                     if k != "loss_history"}
        print(f"  bandwidth x{mult}: img {r['final_img_loss']:.4f}", flush=True)
    # w_off sensitivity
    for wo in (0.0, 0.3):
        cfg2 = {**cfg, "w_off": wo}
        r = fit_once(setup, cfg2, base, tgt, lam, seed=0)
        out[f"w_off_{wo}"] = {k: v for k, v in r.items() if k != "loss_history"}
        print(f"  w_off={wo}: img {r['final_img_loss']:.4f}", flush=True)
    out["_hypothesis"] = key
    suffix = f"_{key}" if hyp else ""
    (OUT / f"sensitivity{suffix}.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["baselines", "orient", "ensemble",
                                        "sensitivity"], default=None)
    ap.add_argument("--sensitivity-hyp", default=None,
                    help="override the orientation hypothesis for sensitivity "
                         "(e.g. flip1_swap1); output gets a suffix")
    args = ap.parse_args()

    for gate in ("register.json", "lambda.json"):
        if not (PRECAL / gate).exists():
            sys.exit(f"GATE FAILURE: {PRECAL / gate} missing — run "
                     "run_stageD_precal.py first (D0 discipline)")
    register = json.loads((PRECAL / "register.json").read_text())
    lam_json = json.loads((PRECAL / "lambda.json").read_text())
    # provenance gate (audit #8): existence is not enough — the precal must
    # have been produced by THIS code and config
    check_provenance(register, what="precal/register.json")
    check_provenance(lam_json, what="precal/lambda.json")
    lam = float(lam_json["lambda_star"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_stage_d(device=device)
    OUT.mkdir(parents=True, exist_ok=True)
    base = conditioning(setup, cfg)
    real = load_targets(cfg, device=device)
    print(f"device={device}  lambda*={lam:g}  V_eff={cfg['voltage_V']/1e6:.3f} MV",
          flush=True)

    # orient FIRST: baselines are now evaluated per orientation (audit #3)
    stages = [args.stage] if args.stage else ["orient", "baselines", "ensemble",
                                              "sensitivity"]
    if "orient" in stages:
        stage_orient(setup, cfg, base, real, lam)
    if "baselines" in stages:
        stage_baselines(setup, cfg, base, real, lam)
    if "ensemble" in stages:
        stage_ensemble(setup, cfg, base, real, lam, register)
    if "sensitivity" in stages:
        stage_sensitivity(setup, cfg, base, real, lam, hyp=args.sensitivity_hyp)
    print(f"done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
