"""Stage-D2 phase 1: reachability fits on E300_15673 with the C_B3_ema CNF.

How close can the 37-knob CNF get to the real images when every knob, the beam
latent and the calibration nuisances are bounded only by the training box and a
weak ridge (λ_reach = λ*_D / 100)? Arms (interview 2026-10-29):

  A1  17 linac/FF δ + 14 injector δ, latent at the prior mean
  A2  A1 + one fitted latent per bunch, shared by the scan
  A3  A2 + a per-step latent residual (prior 0.3 × the prior σ)
  A4  17 linac/FF δ only (injector at golden)    } injector ablations
  A5  17 + the 6 active injector knobs           }
  A6  17 + active 6 + L0B 2                      }
  X1  the xtcav23 flow refit under the same D2 settings (17 δ, no σ_δ/ε_x,
      + L1 shift, λ_reach, full box): the re-baseline
  W1  A1 on the wide-box fine-tune c_b3w_ema (fit-only model, models/c_b3w_ema;
      scripts/stageD2_after_widebox.sh): charge, d2 delay, L1/L2 boxes widened

each under one of three off-shot row-correction conventions (suffix):

  _D  Stage D's: real targets corrected (deployed, contaminated references),
      model images NOT corrected — asymmetric (port review: correcting the
      model alone moves its images by TV ~1.0 at early steps)
  _N  neither side corrected (targets_rowref_none)
  _B  both corrected: clean real references (targets_rowref_clean) + the same
      deployed op on the model's own TCAV-off image (ForwardModel row_center)

Every arm fits ν_y, L2_ref, the L1 shift and δ_tr; σ_δ and ε_x are dropped.
The weak ridge covers the CNF's inputs only (knobs, latent, L1/L2 shifts); ν_y
and δ_tr (measurement chain, no box) keep Stage D's λ* (config lambda_cal) —
at λ_reach the unbounded δ_tr ran to −5…−8% k1 in 3 iterations (2026-10-29).
Light protocol: deterministic loss (latent fixed or fitted, never redrawn),
restart agreement, wall flags. Orientation: `--orient` refits A1_D on all four
row-order × streak-pairing hypotheses.

  cd /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning && \
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/run_stageD2_reach.py \
    --arms A1_N --restarts 3

Outputs -> results/stageD2/15673/reach/<arm>.json (+ orient.json)
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_stageD_fit import CHUNK, ITERS, orient_targets  # noqa: E402
from xtcav_gpsr_tuning.fit import (ForwardModel, eval_loss_chunked,  # noqa: E402
                                   run_fit_chunked)
from xtcav_gpsr_tuning.stage_d import (conditioning, load_stage_d,  # noqa: E402
                                       load_targets, provenance)

CFG_D2 = REPO / "configs" / "fit_stageD2_15673.yaml"
CFG_D2W = REPO / "configs" / "fit_stageD2W_15673.yaml"   # = CFG_D2 on the wide-box bundle
CFG_D = REPO / "configs" / "fit_stageD_15673.yaml"
OUT = REPO / "results" / "stageD2" / "15673" / "reach"

LINAC17 = ["L1EnergyOffset", "L2EnergyOffset", "L3EnergyOffset", "S1ELkG", "S2ELkG",
           "S3ELkG", "Q5FFkG", "Q4FFkG", "Q3FFkG", "S1EL_xOffset", "S1EL_yOffset",
           "S2EL_xOffset", "S2EL_yOffset", "S2ER_xOffset", "S2ER_yOffset",
           "S1ER_xOffset", "S1ER_yOffset"]
ACTIVE6 = ["laser_timing_ps", "GUNF:rf_field_scale", "GUNF:theta0_deg",
           "distgen:t_dist:dists:d2:avg_t:value", "drive_fraction",
           "distgen:total_charge:value"]
L0B2 = ["L0BPhaseSet", "L0BEnergyOffset"]
PASSIVE6 = ["L0AF_phase_offset_deg", "SOL10111:solenoid_field_scale",
            "distgen:r_dist:sigma_xy:value", "distgen:r_dist:truncation_radius:value",
            "distgen:start:MTE:value", "sigma_t_ps"]
ALL31 = LINAC17 + L0B2 + ACTIVE6 + PASSIVE6

CONVENTIONS = {"D": dict(targets="deployed", row_center=False),
               "N": dict(targets="none", row_center=False),
               "B": dict(targets="clean", row_center=True)}
_BASE_ARMS = {
    "A1": dict(model="cnf", knobs=ALL31, latent="mean"),
    "A2": dict(model="cnf", knobs=ALL31, latent="shared"),
    "A3": dict(model="cnf", knobs=ALL31, latent="shared_resid"),
    "A4": dict(model="cnf", knobs=LINAC17, latent="mean"),
    "A5": dict(model="cnf", knobs=LINAC17 + ACTIVE6, latent="mean"),
    "A6": dict(model="cnf", knobs=LINAC17 + ACTIVE6 + L0B2, latent="mean"),
    "X1": dict(model="xtcav23", knobs=LINAC17, latent=None),
    "W1": dict(model="cnf_wide", knobs=ALL31, latent="mean"),
}
ARMS = {f"{a}_{c}": {**spec, **conv} for a, spec in _BASE_ARMS.items()
        for c, conv in CONVENTIONS.items()}
# Stage-D2 follow-ups from the 2026-04-15 elog (convention B):
#   veff_scale — V_eff re-anchored to the dedicated ±90 deg TCAV calibration at 23:53
#     (1219 / 1182 um/deg -> 16.47 um/um) instead of the jitter regression (14.40,
#     biased low by phase-readback noise; per sign +1132 / -944 um/deg)
#   l2_gain — fitted L2 scan gain (the multiknob moves only LI12-14)
CAL_DEDICATED = 0.5 * (1219.0 + 1182.0) / 72.88
VEFF_DEDICATED = CAL_DEDICATED / 14.402902846221519
ARMS.update({
    "A1cal_B": {**ARMS["A1_B"], "veff_scale": VEFF_DEDICATED},
    "A1g_B": {**ARMS["A1_B"], "l2_gain": True},
    "A1calg_B": {**ARMS["A1_B"], "veff_scale": VEFF_DEDICATED, "l2_gain": True},
    "X1cal_B": {**ARMS["X1_B"], "veff_scale": VEFF_DEDICATED},
})
CFGS = {"cnf": CFG_D2, "cnf_wide": CFG_D2W, "xtcav23": CFG_D}
TARGET_FILES = {"deployed": None,  # the Stage D targets.npz (config targets_npz)
                "clean": "results/stageD2/15673/targets_rowref_clean.npz",
                "none": "results/stageD2/15673/targets_rowref_none.npz"}


def subset_setup(setup, names, cfg_d2):
    """Copy of `setup` fitting only `names`, with the D2 σ_j table and bound."""
    s = copy.copy(setup)
    keys = setup.norm["knob_keys"]
    idx = [keys.index(k) for k in names]
    lo = np.array(setup.norm["knob_low"])
    hi = np.array(setup.norm["knob_high"])
    span = (hi - lo)[idx]
    s.fit_knobs = list(names)
    s.knob_index = idx
    s.sigma_norm = torch.tensor([float(cfg_d2["sigma_phys"][k]) / float(sp)
                                 for k, sp in zip(names, span)],
                                dtype=torch.float64, device=setup.device)
    s.bound_norm = torch.full((len(names),), float(cfg_d2["delta_bound_frac"]),
                              dtype=torch.float64, device=setup.device)
    # asymmetric [-lo, +hi] about each knob's baseline u0: bound_frac 0.5 = exactly the box
    u0 = setup.baseline_norm[idx]
    f = float(cfg_d2["delta_bound_frac"]) / 0.5
    s.bound_lo, s.bound_hi = f * u0, f * (1.0 - u0)
    return s


_SETUPS = {}


def get_setup(model, device):
    """(cfg, setup, base) per model, cached: the shared segment is mutated by
    every forward (TDC voltage, δ_tr quads), but always from pristine values."""
    if model not in _SETUPS:
        cfg, setup = load_stage_d(CFGS[model], device=device)
        _SETUPS[model] = (cfg, setup, conditioning(setup, cfg))
    return _SETUPS[model]


def make_model(arm, device, seed=0, rng=None):
    spec = ARMS[arm]
    cfg_d2 = yaml.safe_load(CFG_D2.read_text())
    cfg, setup, base = get_setup(spec["model"], device)
    s = subset_setup(setup, spec["knobs"], cfg_d2)
    scan = None
    if spec.get("l2_gain"):
        mc = json.loads(Path(cfg["machine_config"]).read_text())
        scan = torch.tensor(mc["L2_scan_vals_deg"], dtype=torch.float64)
    m = ForwardModel(s, base, cfg["voltage_V"] * float(spec.get("veff_scale", 1.0)),
                     n_particles=int(cfg["n_particles"]), scan_deg=scan,
                     anchor="none", image_anchor=True, l2ref_nuisance=True,
                     l1ref_nuisance=True, seed=seed, base_off_norm=base,
                     latent_mode=spec["latent"], row_center=spec["row_center"],
                     resid_tau=float(cfg_d2["latent_resid_tau"])).to(device)
    if rng is not None:  # restart jitter, as Stage D (δ raw ~ N(0, 0.3), δ_tr ~ 0.5%)
        with torch.no_grad():
            m.delta_raw.copy_(torch.tensor(rng.normal(0.0, 0.3, size=m.delta_raw.shape)))
            m.dtr.copy_(torch.tensor(rng.normal(0.0, 0.005, size=(3,))))
            if spec["latent"] in ("shared", "shared_resid"):  # prior-σ units (port review #2/#3)
                m.s_lat.add_(torch.as_tensor(rng.normal(0.0, 0.5, size=tuple(m.s_lat.shape)),
                                             dtype=m.s_lat.dtype, device=m.s_lat.device))
    return m, cfg, cfg_d2


def evaluate(m, tgt, cfg):
    return eval_loss_chunked(m, tgt["images"], tgt["splits"], mask=tgt["mask"],
                             step_weights=tgt["weights"], targets_off=tgt["off"],
                             off_step_weights=tgt.get("off_weights"),
                             w_off=float(cfg.get("w_off", 0.1)), chunk=2 * CHUNK)


STEP_SETS = {"even": lambda i: i % 2 == 0, "odd": lambda i: i % 2 == 1,
             "low": lambda i: i <= 16, "deep": lambda i: i >= 17}


def holdout(tgt, fit_steps):
    """(fit targets, held-out step indices): the held-out steps' image and off weights
    are zeroed for the fit (Phase 2 split tests); 'all' returns tgt unchanged."""
    if fit_steps == "all":
        return tgt, []
    B = tgt["weights"].shape[1]
    keep = torch.tensor([STEP_SETS[fit_steps](i) for i in range(B)], device=tgt["weights"].device)
    out = dict(tgt)
    out["weights"] = tgt["weights"] * keep
    out["off_weights"] = tgt["off_weights"] * keep
    return out, [i for i in range(B) if not bool(keep[i])]


def fit_once(arm, tgt, device, seed=0, rng=None, iters=ITERS, lam=None, fit_steps="all"):
    m, cfg, cfg_d2 = make_model(arm, device, seed=seed, rng=rng)
    tgt_eval = tgt
    tgt, held = holdout(tgt, fit_steps)
    lam = float(cfg_d2["lambda_reach"]) if lam is None else lam
    null = evaluate(m, tgt_eval, cfg) if rng is None else None
    t0 = time.time()
    res = run_fit_chunked(
        m, tgt["images"], tgt["splits"], lam=lam, iters=iters, lr=0.04,
        w_q=float(cfg.get("w_charge_split", 0.1)), chunk=CHUNK, mask=tgt["mask"],
        step_weights=tgt["weights"], targets_off=tgt["off"],
        w_off=float(cfg.get("w_off", 0.1)), prior="ridge",
        l2ref_sigma_deg=float(cfg["l2ref_sigma_deg"]),
        nu_y_sigma_ln=cfg.get("nu_y_prior_sigma"),
        dtr_sigma_frac=cfg.get("dtr_prior_sigma_frac"),
        off_step_weights=tgt.get("off_weights"),
        l1ref_sigma_deg=float(cfg_d2["l1ref_sigma_deg"]),
        lam_cal=float(cfg_d2["lambda_cal"]))
    parts = evaluate(m, tgt_eval, cfg)   # all steps, real weights (held-out steps included)
    if held:
        pt = parts["per_image_tv"]
        we = tgt_eval["weights"].cpu().numpy()
        fitted = [i for i in range(pt.shape[1]) if i not in held]
        res["holdout"] = {"fit_steps": fit_steps, "held_steps": held,
                          "img_fit": float((pt[:, fitted] * we[:, fitted]).mean()),
                          "img_held": float((pt[:, held] * we[:, held]).mean())}
        if null is not None:
            pn = null["per_image_tv"]
            res["holdout"].update(null_fit=float((pn[:, fitted] * we[:, fitted]).mean()),
                                  null_held=float((pn[:, held] * we[:, held]).mean()))
    res.update(final_img_loss=parts["img"], final_off_loss=parts["off"],
               final_split_loss=parts["split"], per_image_tv=parts["per_image_tv"].tolist(),
               l2_shift_deg=float(m.l2_shift_deg), seed=int(seed), lam=lam, iters=iters,
               total_loss=float(res["loss_history"][-1]), fit_seconds=time.time() - t0)
    if null is not None:
        res["null"] = {"img": null["img"], "off": null["off"], "split": null["split"],
                       "per_image_tv": null["per_image_tv"].tolist()}
    if m.latent_mode == "shared_resid":
        res["s_resid"] = m.s_resid.detach().cpu().tolist()
    res["state"] = {k: v.detach().cpu().tolist() for k, v in m.state_dict().items()
                    if k in ("delta_raw", "dtr", "log_nu_y", "l2_shift_deg", "l1_shift_deg",
                             "s_lat", "s_resid")}
    return res


def load_real(device, variant="deployed"):
    cfg, _, _ = get_setup("cnf", device)
    if TARGET_FILES[variant] is not None:
        cfg = {**cfg, "targets_npz": TARGET_FILES[variant]}
    return load_targets(cfg, device=device)


def run_orient(device, iters):
    real = load_real(device)
    runs = {}
    for fr in (False, True):
        for sw in (False, True):
            key = f"flip{int(fr)}_swap{int(sw)}"
            r = fit_once("A1_D", orient_targets(real, fr, sw), device, iters=iters)
            runs[key] = {"flip_rows": fr, "swap_signs": sw, "img_loss": r["final_img_loss"],
                         "null_img": r["null"]["img"], "total": r["total_loss"]}
            print(f"  orient {key}: null {r['null']['img']:.4f} -> fit {r['final_img_loss']:.4f}",
                  flush=True)
    ranked = sorted(runs, key=lambda k: runs[k]["img_loss"])
    sel = {"runs": runs, "ranked": ranked, "winner": ranked[0],
           "runner_up_ratio": runs[ranked[1]]["img_loss"] / runs[ranked[0]]["img_loss"],
           "provenance": provenance(CFG_D2)}
    (OUT / "orient.json").write_text(json.dumps(sel, indent=1))
    print(f"  winner {ranked[0]} (ratio {sel['runner_up_ratio']:.2f})", flush=True)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--start", type=int, default=0,
                    help="first restart index; >0 appends to the arm's existing json")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--orient", action="store_true")
    ap.add_argument("--hyp", default=None, help="flipF_swapS; default: orient.json winner")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fit-steps", default="all", choices=["all"] + list(STEP_SETS),
                    help="fit on a step subset; held-out steps are scored separately")
    ap.add_argument("--lam", type=float, default=None,
                    help="knob/latent ridge λ (default: the config's lambda_reach); ν_y / δ_tr keep "
                         "lambda_cal")
    ap.add_argument("--targets-file", default=None,
                    help="targets npz replacing the arm's real targets (e.g. S2E truth targets)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.orient:
        run_orient(device, args.iters)
    if not args.arms:
        return
    hyp = args.hyp or json.loads((OUT / "orient.json").read_text())["winner"]
    fr, sw = hyp == "flip1_swap1" or hyp.startswith("flip1"), hyp.endswith("swap1")
    for arm in args.arms:
        if args.targets_file:
            cfg0, _, _ = get_setup("cnf", device)
            tgt = orient_targets(load_targets({**cfg0, "targets_npz": args.targets_file},
                                              device=device), fr, sw)
        else:
            tgt = orient_targets(load_real(device, ARMS[arm]["targets"]), fr, sw)
        runs, hist = [], []
        prev = OUT / f"{arm}{args.tag}.json"
        if args.start > 0 and prev.exists():
            d = json.loads(prev.read_text())
            runs, hist = d["runs"], d["loss_history"]
            runs = [r for r in runs if r["seed"] < args.start]
            hist = hist[:len(runs)]
        for s in range(args.start, args.start + args.restarts):
            rng = np.random.default_rng(1000 + s) if s else None  # restart 0 = zero init
            r = fit_once(arm, tgt, device, seed=s, rng=rng, iters=args.iters,
                         fit_steps=args.fit_steps, lam=args.lam)
            hist.append(r.pop("loss_history"))
            runs.append(r)
            print(f"  {arm} restart {s}: img {r['final_img_loss']:.4f} off {r['final_off_loss']:.4f} "
                  f"split {r['final_split_loss']:.4f} walls {r['walls']} "
                  f"({r['fit_seconds'] / 60:.1f} min)", flush=True)
            out = {"arm": arm, "spec": {k: v for k, v in ARMS[arm].items()}, "hypothesis": hyp,
                   "runs": runs, "loss_history": hist,
                   "provenance": provenance(CFGS[ARMS[arm]["model"]])}
            (OUT / f"{arm}{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
