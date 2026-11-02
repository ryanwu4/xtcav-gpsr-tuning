"""Stage D3: fit the three 2026-04-15 E300 L2 scans with the wide-box CNF, on trusted L2 windows.

Ryan's design (2026-11-01, memory project_stageD3_multirun): the Stage D2 best arm (W1calgr_B:
31 knobs, latent at the prior mean, convention B row correction, dedicated-cal V_eff, fitted
L2 scan gain, fitted resolution) with everything shared across runs (injector incl. ONE cathode
charge, linac, FF, movers, V_eff, ν_y, δ_tr, resolution, one L2 gain) except a per-run L1 phase
(starts at the klystron readback, prior σ L1_SIGMA_DEG) and a per-run L2 reference (starts at
the July sim2real alignment, prior σ L2_SIGMA_DEG). Steps: each run's window from
results/stageD3/step_selection.json ('win') or all 33 steps ('full').

Arms (each fit is scored afterwards on the FULL scans of its runs, so the excluded deep steps
are a held-out residual):

  S<run>_win / S<run>_full   one run alone
  J_win / J_full              all three runs jointly
  J_win_drift                 J_win with the recorded per-step L1 deltas (BC11 feedback drift)
  N_full                      no fit: golden injector + linac at the readback L1 / July L2 starts,
                              scored on all three full scans (the null)
  LORO<run>                   J_win on the other two runs, then ONLY the held-out run's L1 and
                              L2 shifts refit on its window (everything shared frozen), scored
                              on its full scan: does one shared machine state transfer?

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src ~/miniconda3/envs/slac-rl/bin/python \\
    scripts/run_stageD3_fit.py --arms J_win --restarts 2

-> results/stageD3/fits/<arm>.json, results/stageD3/render/<arm>.npz (best restart, full scans)
"""

import argparse
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
from run_stageD2_reach import (ALL31, CFG_D2, VEFF_DEDICATED, get_setup,  # noqa: E402
                               subset_setup)
from run_stageD_fit import CHUNK, ITERS, orient_targets  # noqa: E402
from xtcav_gpsr_tuning.fit import (ForwardModel, eval_loss_chunked,  # noqa: E402
                                   run_fit_chunked)
from xtcav_gpsr_tuning.stage_d import conditioning, load_targets, provenance  # noqa: E402

OUT = REPO / "results" / "stageD3"
MC = "/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results/tier1_sim_scans/{}/machine_config.json"
# L1 anchor = the klystron-readback anchor of each machine_config (15668: LI11 PACT median;
# 15671/15673: the dataset comment, confirmed by PACT to 0.2 deg). L2 start = the July sim2real
# alignment at THAT L1 (15671: the phase-anchored -0.35 deg, not the sigma-anchored +1.95 deg,
# which belongs to L1 -22.35; 15668: the l2ref-40.35 rerun; 15673: the D2 config value).
RUNS = {
    "15668": dict(dataset="TEST_15668", l2_ref=-40.35,
                  targets="results/stageD3/15668/targets_rowref_clean.npz"),
    "15671": dict(dataset="E300_15671", l2_ref=-38.35,
                  targets="results/stageD3/15671/targets_rowref_clean.npz"),
    "15673": dict(dataset="E300_15673", l2_ref=-37.9,
                  targets="results/stageD2/15673/targets_rowref_clean.npz"),
}
ORDER = ("15668", "15671", "15673")
HYP = (True, True)        # flip rows, swap signs: the D2 orientation winner (flip1_swap1), same camera
L1_SIGMA_DEG = 2.0        # Ryan: readback start, wide prior; the fitted inter-run gaps are a result
L2_SIGMA_DEG = 1.0        # the July alignment used the 2024 golden injector beam (fit-choice spread ~1.5 deg)
MODEL = "cnf_wide"        # models/c_b3w_ema


def arm_spec(arm):
    """(runs, step mode, drift, held-out run or None) of an arm name."""
    if arm.startswith("LORO"):
        out = arm[4:]
        return tuple(r for r in ORDER if r != out), "win", False, out
    if arm.startswith("S"):
        run, mode = arm[1:].split("_")
        return (run,), mode, False, None
    parts = arm.split("_")
    return ORDER, parts[1], "drift" in parts, None


def windows():
    sel = json.loads((OUT / "step_selection.json").read_text())
    return {r: sel[RUNS[r]["dataset"]]["kept_steps"] for r in ORDER}


def l1_deltas(run):
    d = json.loads(Path(MC.format(RUNS[run]["dataset"])).read_text())["L1_delta_per_step_deg"]
    return np.array([float(d[str(i)]) for i in range(1, len(d) + 1)])


def build(runs, mode, drift, device):
    """Concatenated conditioning rows, run index, scan offsets and oriented targets."""
    cfg, setup, _ = get_setup(MODEL, device)
    win = windows() if mode == "win" else None
    bases, ridx, scans, tg, meta = [], [], [], [], []
    for j, r in enumerate(runs):
        mc = json.loads(Path(MC.format(RUNS[r]["dataset"])).read_text())
        cfg_r = {**cfg, "machine_config": MC.format(RUNS[r]["dataset"]),
                 "l1_anchor_deg": float(mc["L1_anchor_deg"]), "l2_ref_deg": RUNS[r]["l2_ref"]}
        base = conditioning(setup, cfg_r, l1_deltas(r) if drift else None)
        steps = list(range(base.shape[0])) if win is None else win[r]
        raw = load_targets({**cfg, "targets_npz": RUNS[r]["targets"]}, device=device)
        t = {**orient_targets(raw, *HYP), "off_floors": raw["off_floors"]}
        bases.append(base[steps])
        ridx += [j] * len(steps)
        scans.append(torch.tensor(mc["L2_scan_vals_deg"], dtype=torch.float64)[steps])
        tg.append({"images": t["images"][:, steps], "splits": t["splits"][:, steps],
                   "floors": t["floors"][:, steps], "off": t["off"][steps],
                   "off_floors": t["off_floors"][steps], "mask": t["mask"]})
        meta += [{"run": r, "step": int(s), "scan_deg": float(mc["L2_scan_vals_deg"][s])} for s in steps]
    tgt = {"images": torch.cat([t["images"] for t in tg], 1),
           "splits": torch.cat([t["splits"] for t in tg], 1),
           "floors": torch.cat([t["floors"] for t in tg], 1),
           "off": torch.cat([t["off"] for t in tg], 0),
           "mask": tg[0]["mask"]}
    off_floors = torch.cat([t["off_floors"] for t in tg], 0)
    # noise weights over the whole batch (load_targets' rule, one median for all runs)
    fmed = float(torch.nanmedian(tgt["floors"]))
    tgt["weights"] = (fmed / tgt["floors"].clamp_min(fmed)).nan_to_num(0.0)
    ofmed = float(torch.nanmedian(off_floors))
    tgt["off_weights"] = (ofmed / off_floors.clamp_min(ofmed)).nan_to_num(0.0)
    return cfg, setup, torch.cat(bases, 0), torch.tensor(ridx), torch.cat(scans), tgt, meta


def make_model(cfg, setup, base, ridx, scan, device, seed=0, rng=None):
    cfg_d2 = yaml.safe_load(CFG_D2.read_text())
    s = subset_setup(setup, ALL31, cfg_d2)
    m = ForwardModel(s, base, cfg["voltage_V"] * VEFF_DEDICATED, n_particles=int(cfg["n_particles"]),
                     scan_deg=scan, anchor="none", image_anchor=True, l2ref_nuisance=True,
                     l1ref_nuisance=True, seed=seed, base_off_norm=base, latent_mode="mean",
                     row_center=True, fit_blur=True, run_idx=ridx.to(device)).to(device)
    if rng is not None:  # restart jitter, as Stage D / D2
        with torch.no_grad():
            m.delta_raw.copy_(torch.tensor(rng.normal(0.0, 0.3, size=m.delta_raw.shape)))
            m.dtr.copy_(torch.tensor(rng.normal(0.0, 0.005, size=(3,))))
            m.l1_shift_deg.copy_(torch.tensor(rng.normal(0.0, 0.5, size=m.l1_shift_deg.shape)))
    return m, cfg_d2


def fit(m, tgt, cfg, cfg_d2, iters):
    return run_fit_chunked(
        m, tgt["images"], tgt["splits"], lam=float(cfg_d2["lambda_reach"]), iters=iters, lr=0.04,
        w_q=float(cfg.get("w_charge_split", 0.1)), chunk=CHUNK, mask=tgt["mask"],
        step_weights=tgt["weights"], targets_off=tgt["off"], w_off=float(cfg.get("w_off", 0.1)),
        prior="ridge", l2ref_sigma_deg=L2_SIGMA_DEG, nu_y_sigma_ln=cfg.get("nu_y_prior_sigma"),
        dtr_sigma_frac=cfg.get("dtr_prior_sigma_frac"), off_step_weights=tgt["off_weights"],
        l1ref_sigma_deg=L1_SIGMA_DEG, lam_cal=float(cfg_d2["lambda_cal"]))


def evaluate(m, tgt, cfg):
    return eval_loss_chunked(m, tgt["images"], tgt["splits"], mask=tgt["mask"],
                             step_weights=tgt["weights"], targets_off=tgt["off"],
                             off_step_weights=tgt["off_weights"], w_off=float(cfg.get("w_off", 0.1)),
                             chunk=2 * CHUNK)


def state(m):
    return {k: v.detach().cpu().tolist() for k, v in m.state_dict().items()
            if k in ("delta_raw", "dtr", "log_nu_y", "l2_shift_deg", "l1_shift_deg", "l2_gain_m1", "blur_u")}


def load_state(m, st, runs_from=None, runs_to=None):
    """Load a fit state; with runs_from/runs_to, the per-run shifts are re-indexed (LORO and the
    full-scan scoring of a subset fit)."""
    sd = {k: torch.tensor(v, dtype=torch.float64) for k, v in st.items()}
    if runs_from is not None:
        for k in ("l1_shift_deg", "l2_shift_deg"):
            src = sd[k]
            sd[k] = torch.stack([src[runs_from.index(r)] if r in runs_from else torch.zeros((), dtype=torch.float64)
                                 for r in runs_to])
    bad = m.load_state_dict(sd, strict=False)
    assert not bad.unexpected_keys, bad.unexpected_keys


@torch.no_grad()
def score_full(st, runs, drift, device, fit_runs=None):
    """Render the FULL scans of `runs` at a fitted state: per-image TV, floors, renders."""
    cfg, setup, base, ridx, scan, tgt, meta = build(runs, "full", drift, device)
    m, _ = make_model(cfg, setup, base, ridx, scan, device)
    load_state(m, st, fit_runs or list(runs), list(runs))
    parts = evaluate(m, tgt, cfg)
    imgs = []
    for lo in range(0, base.shape[0], 2 * CHUNK):
        rows = list(range(lo, min(lo + 2 * CHUNK, base.shape[0])))
        imgs.append(m.forward_full(rows=rows)[0].float().cpu())
    return {"per_image_tv": parts["per_image_tv"].tolist(), "floors": tgt["floors"].cpu().tolist(),
            "weights": tgt["weights"].cpu().tolist(), "meta": meta}, torch.cat(imgs, 1).numpy()


def summarize(m, res, runs):
    l1 = m.l1_shift_deg.detach().cpu().numpy().tolist()
    l2 = m.l2_shift_deg.detach().cpu().numpy().tolist()
    res["per_run"] = {r: {"l1_shift_deg": l1[j], "l2_shift_deg": l2[j],
                          "l1_deg": float(json.loads(Path(MC.format(RUNS[r]["dataset"])).read_text())
                                          ["L1_anchor_deg"]) + l1[j],
                          "l2_ref_deg": RUNS[r]["l2_ref"] + l2[j]} for j, r in enumerate(runs)}
    return res


def run_null(device):
    cfg, setup, base, ridx, scan, tgt, meta = build(ORDER, "full", False, device)
    m, _ = make_model(cfg, setup, base, ridx, scan, device)
    sc, imgs = score_full(state(m), ORDER, False, device)
    out = {"arm": "N_full", "runs": list(ORDER), "score_full": {**sc, "seed": 0}}
    (OUT / "fits").mkdir(parents=True, exist_ok=True)
    (OUT / "fits" / "N_full.json").write_text(json.dumps(out, indent=1))
    (OUT / "render").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "render" / "N_full.npz", imgs=imgs.astype(np.float32),
                        meta=np.array(json.dumps(sc["meta"])))
    tv = np.asarray(sc["per_image_tv"]).mean(0)
    for r in ORDER:
        rows = [i for i, q in enumerate(sc["meta"]) if q["run"] == r]
        print(f"  N_full TV {r}: " + " ".join(f"{tv[i]:.2f}" for i in rows), flush=True)


def run_arm(arm, restarts, iters, device):
    if arm == "N_full":
        return run_null(device)
    runs, mode, drift, held = arm_spec(arm)
    cfg, setup, base, ridx, scan, tgt, meta = build(runs, mode, drift, device)
    out = {"arm": arm, "runs": list(runs), "mode": mode, "drift": drift, "held_out": held,
           "model": MODEL, "l1_sigma_deg": L1_SIGMA_DEG, "l2_sigma_deg": L2_SIGMA_DEG,
           "meta": meta, "fits": [], "provenance": provenance(REPO / "configs" / "fit_stageD2W_15673.yaml")}
    path = OUT / "fits" / f"{arm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    best, best_loss = None, np.inf
    for s in range(restarts):
        rng = np.random.default_rng(1000 + s) if s else None
        m, cfg_d2 = make_model(cfg, setup, base, ridx, scan, device, seed=s, rng=rng)
        t0 = time.time()
        res = fit(m, tgt, cfg, cfg_d2, iters)
        hist = res.pop("loss_history")
        parts = evaluate(m, tgt, cfg)
        res.update(seed=s, final_img_loss=parts["img"], final_off_loss=parts["off"],
                   final_split_loss=parts["split"], per_image_tv=parts["per_image_tv"].tolist(),
                   loss_last=hist[-1], fit_seconds=time.time() - t0, state=state(m))
        summarize(m, res, runs)
        out["fits"].append(res)
        print(f"  {arm} restart {s}: img {parts['img']:.4f} off {parts['off']:.4f} g {res['l2_gain']:.3f} "
              f"L1 {[round(v['l1_deg'], 2) for v in res['per_run'].values()]} walls {res['walls']} "
              f"({res['fit_seconds'] / 60:.1f} min)", flush=True)
        if parts["img"] < best_loss:
            best, best_loss = res, parts["img"]
        path.write_text(json.dumps(out, indent=1))
    if held is not None:  # LORO: refit only the held-out run's L1/L2 shifts on its window
        cfg, setup, base_h, ridx_h, scan_h, tgt_h, meta_h = build((held,), "win", drift, device)
        m, cfg_d2 = make_model(cfg, setup, base_h, ridx_h, scan_h, device)
        load_state(m, best["state"], list(runs), [held])
        for n, p in m.named_parameters():
            p.requires_grad_(n in ("l1_shift_deg", "l2_shift_deg"))
        null = evaluate(m, tgt_h, cfg)
        res = fit(m, tgt_h, cfg, cfg_d2, max(40, iters // 2))
        res.pop("loss_history")
        parts = evaluate(m, tgt_h, cfg)
        res.update(img_before_refit=null["img"], final_img_loss=parts["img"],
                   per_image_tv=parts["per_image_tv"].tolist(), meta=meta_h, state=state(m))
        summarize(m, res, (held,))
        out["held_out_fit"] = res
        print(f"  {arm} held-out {held}: window img {null['img']:.4f} (shared state, own L1/L2 at the "
              f"prior start) -> {parts['img']:.4f} after the 2-shift refit; "
              f"L1 {res['per_run'][held]['l1_deg']:.2f}", flush=True)
        path.write_text(json.dumps(out, indent=1))
        sc, imgs = score_full(res["state"], (held,), drift, device)
    else:
        sc, imgs = score_full(best["state"], runs, drift, device)
    out["score_full"] = {**sc, "seed": best["seed"]}
    path.write_text(json.dumps(out, indent=1))
    (OUT / "render").mkdir(exist_ok=True)
    np.savez_compressed(OUT / "render" / f"{arm}.npz", imgs=imgs.astype(np.float32),
                        meta=np.array(json.dumps(sc["meta"])))
    tv = np.asarray(sc["per_image_tv"]).mean(0)
    for r in (runs if held is None else (held,)):
        rows = [i for i, q in enumerate(sc["meta"]) if q["run"] == r]
        print(f"  {arm} full-scan TV {r}: " + " ".join(f"{tv[i]:.2f}" for i in rows), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for arm in args.arms:
        t0 = time.time()
        run_arm(arm, args.restarts, args.iters, device)
        print(f"{arm} done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
