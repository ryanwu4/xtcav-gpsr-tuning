"""Stage-D D4 figures for the E300_15673 fit.

Rebuilds the best converged MAP model per carried hypothesis from ensemble.json
and renders: target-vs-fit gallery, per-step TV overlay, pulls/projections
chart, separation curve, off-spot check. Writes a numbers.json used by REPORT.md.

  cd /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning && \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=$PWD/src:/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/stageD_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from run_stageD_fit import make_model, orient_targets  # noqa: E402
from xtcav_gpsr_tuning.screen import PIXEL_UM  # noqa: E402
from xtcav_gpsr_tuning.stage_d import (conditioning, load_stage_d,  # noqa: E402
                                       load_targets)

FIT = REPO / "results" / "stageD" / "15673" / "fit"
OUT = REPO / "results" / "stageD" / "15673" / "figures"
GALLERY_STEPS = [2, 16, 30]  # scan +0.50, +4.00, +7.50 deg (0-indexed rows)
VMAX_PCT = 99.5  # display: per-panel percentile cap. Sharp (1-px) renders have a
                 # far higher peak than the smooth 14-shot-mean targets, so raw
                 # autoscale makes the model look artificially faint — a display
                 # artifact, not a fit property.
DIAG_SMOOTH_PX = 3.0  # separation/off diagnostics only: both sides get the SAME
                      # Gaussian smoothing first. The two-lobe COM-split estimator
                      # is not scale-free — on a sharp filament the split lands
                      # inside the drive lobe, on a smooth image it doesn't, so
                      # comparing unsmoothed sharp-model vs smooth-real silently
                      # breaks the same-estimator principle.


def set_model_params(model, run):
    with torch.no_grad():
        pulls = torch.tensor([run["pulls"][k] for k in model.s.fit_knobs],
                             dtype=torch.float64, device=model.base.device)
        dn = pulls * model.s.sigma_norm
        model.delta_raw.copy_(torch.atanh((dn / model.s.bound_norm).clamp(-0.99, 0.99)))
        model.dtr.copy_(torch.tensor(run["dtr"], dtype=torch.float64,
                                     device=model.base.device))
        model.log_nu_y.copy_(torch.tensor(np.log(run["nu_y"]), dtype=torch.float64))
        model.l2_shift_deg.copy_(torch.tensor(run["l2_shift_deg"],
                                              dtype=torch.float64))
        model.log_sdelta.copy_(torch.tensor(np.log(run.get("sdelta", 1.0)),
                                            dtype=torch.float64))
        model.log_ex.copy_(torch.tensor(np.log(run.get("ex", 1.0)),
                                        dtype=torch.float64))


def render_all(model, chunk=6):
    """Streaked + off images via the combined pass (one flow sample per chunk)."""
    B = model.base.shape[0]
    imgs, offs = [], []
    with torch.no_grad():
        for lo in range(0, B, chunk):
            rows = list(range(lo, min(lo + chunk, B)))
            im, _, off = model.forward_full(rows=rows)
            imgs.append(im.cpu())
            offs.append(off.cpu())
    return torch.cat(imgs, dim=1).numpy(), torch.cat(offs, dim=0).numpy()


def _smooth(imgs, sigma_px=DIAG_SMOOTH_PX):
    from scipy.ndimage import gaussian_filter
    a = np.asarray(imgs)
    return np.stack([gaussian_filter(im, sigma_px) for im in a]) if a.ndim == 3 \
        else gaussian_filter(a, sigma_px)


def x_separation_um(imgs):
    """Two-lobe separation on the streak axis: distance between the charge
    centroids of the two sides of the x-profile COM (same estimator for
    target and model images). imgs: (B, ny, nx)."""
    imgs = _smooth(imgs)
    x = np.arange(imgs.shape[-1], dtype=float)
    out = []
    for img in imgs:
        prof = img.sum(axis=-2)
        com = (prof * x).sum() / prof.sum()
        lo, hi = x <= com, x > com
        c_lo = (prof[lo] * x[lo]).sum() / prof[lo].sum()
        c_hi = (prof[hi] * x[hi]).sum() / prof[hi].sum()
        out.append(c_hi - c_lo)
    return np.asarray(out) * PIXEL_UM


def sigmas_um(img):
    img = _smooth(img)
    x = np.arange(img.shape[-1], dtype=float)
    y = np.arange(img.shape[-2], dtype=float)
    px = img.sum(axis=-2) / img.sum(axis=(-2, -1), keepdims=True)[..., 0]
    py = img.sum(axis=-1) / img.sum(axis=(-2, -1), keepdims=True)[..., 0]
    mx = (px * x).sum(-1); my = (py * y).sum(-1)
    sx = np.sqrt((px * (x - mx[..., None]) ** 2).sum(-1)) * PIXEL_UM
    sy = np.sqrt((py * (y - my[..., None]) ** 2).sum(-1)) * PIXEL_UM
    return sx, sy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_stage_d(device=device)
    base = conditioning(setup, cfg)
    real = load_targets(cfg, device=device)
    ens = json.loads((FIT / "ensemble.json").read_text())
    orient_sel = json.loads((FIT / "orient.json").read_text())
    baselines = json.loads((FIT / "baselines.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    scan = real["scan_vals_deg"]
    numbers = {"hypotheses": {}}

    rendered = {}
    for hyp, h in ens["hypotheses"].items():
        # audit #4/#9: use the index the ensemble RECORDED (chosen by the MAP
        # objective) and the seed it was fitted with, instead of re-deriving a
        # different "best" by image loss alone with a defaulted seed.
        ix = h.get("converged_indices", list(range(len(h["runs"]))))
        conv = [h["runs"][i] for i in ix]
        best = h["runs"][h["best_restart"]] if "best_restart" in h \
            else min(conv, key=lambda r: r["final_img_loss"])
        fr = orient_sel["runs"][hyp]["flip_rows"]
        sw = orient_sel["runs"][hyp]["swap_signs"]
        tgt = orient_targets(real, fr, sw)
        if "seed" not in best:
            raise KeyError("ensemble runs carry no 'seed'; re-run the fit stage "
                           "(audit #9) — re-rendering with a defaulted seed uses "
                           "a different frozen CRN draw than the fit")
        model = make_model(setup, cfg, base, seed=int(best["seed"]))
        set_model_params(model, best)
        fit_imgs, fit_off = render_all(model)
        # null (delta = 0)
        null_model = make_model(setup, cfg, base, seed=0)
        null_imgs, null_off = render_all(null_model)
        rendered[hyp] = dict(best=best, tgt=tgt, fit=fit_imgs, off=fit_off,
                             null=null_imgs, null_off=null_off,
                             conv=conv, fr=fr, sw=sw)
        proj = np.array([r["projections"] for r in conv])
        numbers["hypotheses"][hyp] = {
            "n_converged": len(conv), "n_runs": len(h["runs"]),
            "best_img_loss": best["final_img_loss"],
            "best_off_loss": best["final_off_loss"],
            "nu_y": best["nu_y"], "l2_shift_deg": best["l2_shift_deg"],
            "dtr": best["dtr"], "pulls": best["pulls"],
            "delta_phys": best["delta_phys"],
            "proj_median": np.median(proj, 0).tolist(),
            "proj_spread_converged": np.ptp(proj, 0).tolist(),
        }

    main_hyp = min(numbers["hypotheses"],
                   key=lambda k: numbers["hypotheses"][k]["best_img_loss"])
    numbers["main_hypothesis"] = main_hyp
    # audit #3: baselines are keyed by orientation; use the adopted one
    nulls = baselines["null"]
    numbers["null_img_loss"] = (nulls[main_hyp]["img_loss"]
                                if main_hyp in nulls else nulls["img_loss"])
    numbers["null_img_loss_all_orientations"] = {
        k: v["img_loss"] for k, v in nulls.items()} if main_hyp in nulls else None
    nuis = baselines["nuisance_only"]
    numbers["nuisance_img_loss"] = (nuis[main_hyp]["final_img_loss"]
                                    if main_hyp in nuis else nuis["final_img_loss"])
    numbers["floor_tv_median"] = float(torch.nanmedian(real["floors"]))
    R = rendered[main_hyp]

    # ---------------- fig 1: gallery ---------------------------------------- #
    fig, axes = plt.subplots(3, len(GALLERY_STEPS), figsize=(13, 8))
    for c, si in enumerate(GALLERY_STEPS):
        rows = [("real target", R["tgt"]["images"][0, si].cpu().numpy()),
                ("MAP fit", R["fit"][0, si]),
                (r"$\delta=0$ null", R["null"][0, si])]
        for r, (label, im) in enumerate(rows):
            ax = axes[r, c]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis",
                      vmin=0.0, vmax=np.percentile(im, VMAX_PCT))
            if r == 0:
                ax.set_title(f"scan {scan[si]:+.2f}°", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"E300_15673 targets vs fit ({main_hyp}, + streak sign) — "
                 f"each panel scaled to its own {VMAX_PCT}th pct", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gallery.png", dpi=130)
    plt.close(fig)

    # ---------------- fig 2: per-step TV ------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    null_tv = np.array((nulls[main_hyp] if main_hyp in nulls
                        else nulls)["per_image_tv"])
    floors = R["tgt"]["floors"].cpu().numpy()   # audit #3: pairing-swapped
    for gi, sign in enumerate(("+", "−")):
        ax = axes[gi]
        ax.plot(scan, null_tv[gi], "o-", color="0.6", label="δ=0 null")
        for hyp, style in zip(rendered, ("C0", "C1")):
            best_tv = np.array(rendered[hyp]["best"]["per_image_tv"])
            ax.plot(scan, best_tv[gi], "o-", color=style, ms=3,
                    label=f"MAP fit ({hyp})")
        ax.plot(scan, floors[gi], "k--", lw=1, label="split-half floor")
        ax.set_xlabel("L2 scan value [deg]")
        ax.set_title(f"streak sign {sign}")
        if gi == 0:
            ax.set_ylabel("per-image TV (unit-sum)")
            ax.legend(fontsize=8)
    fig.suptitle("Image residual per step: fit vs null vs noise floor", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "per_step_tv.png", dpi=130)
    plt.close(fig)

    # ---------------- fig 3: pulls + projections ---------------------------- #
    reg = json.loads((REPO / "results/stageD/15673/precal/register.json").read_text())
    knobs = list(R["best"]["pulls"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    xpos = np.arange(len(knobs))
    width = 0.38
    for k, (hyp, color) in enumerate(zip(rendered, ("C0", "C1"))):
        conv = rendered[hyp]["conv"]
        med = [np.median([r["pulls"][kb] for r in conv]) for kb in knobs]
        lo = [min(r["pulls"][kb] for r in conv) for kb in knobs]
        hi = [max(r["pulls"][kb] for r in conv) for kb in knobs]
        ax.bar(xpos + (k - 0.5) * width, med, width, color=color, label=hyp,
               yerr=[np.subtract(med, lo), np.subtract(hi, med)],
               error_kw={"lw": 0.8}, alpha=0.85)
    ax.axhline(0, color="k", lw=0.5)
    for yy in (-1, 1):
        ax.axhline(yy, color="0.7", lw=0.6, ls=":")
    ax.set_xticks(xpos)
    ax.set_xticklabels(knobs, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"pull $\hat\delta_j/\sigma_j$")
    ax.legend(fontsize=8)
    ax.set_title("MAP pulls (median ± range over converged restarts)")
    ax = axes[1]
    for k, (hyp, color) in enumerate(zip(rendered, ("C0", "C1"))):
        n = numbers["hypotheses"][hyp]
        med, spr = np.array(n["proj_median"]), np.array(n["proj_spread_converged"])
        xs = np.arange(len(med)) + (k - 0.5) * 0.3
        ax.errorbar(xs, med, yerr=spr / 2, fmt="o", ms=4, color=color, lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(numbers["hypotheses"][main_hyp]["proj_median"])))
    ax.set_xlabel("registered SVD direction")
    ax.set_ylabel("projection [prior-scaled]")
    ax.set_title("Identifiable-subspace projections")
    fig.tight_layout()
    fig.savefig(OUT / "pulls_projections.png", dpi=130)
    plt.close(fig)
    numbers["register_note"] = {"n_kept": reg["n_kept"]}

    # ---------------- fig 4: separation curve ------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    tgt_imgs = R["tgt"]["images"].cpu().numpy()
    sep = {}
    for gi, sign in enumerate(("+", "−")):
        ax = axes[gi]
        s_t = x_separation_um(tgt_imgs[gi])
        s_f = x_separation_um(R["fit"][gi])
        s_n = x_separation_um(R["null"][gi])
        sep[sign] = {"target": s_t.tolist(), "fit": s_f.tolist(), "null": s_n.tolist()}
        ax.plot(scan, s_t, "ko-", ms=3, label="real target")
        ax.plot(scan, s_f, "C0o-", ms=3, label="MAP fit")
        ax.plot(scan, s_n, "o-", color="0.6", ms=3, label="δ=0 null")
        ax.set_xlabel("L2 scan value [deg]")
        ax.set_title(f"streak sign {sign}")
        if gi == 0:
            ax.set_ylabel("two-lobe x separation [µm]")
            ax.legend(fontsize=8)
    fig.suptitle("Drive–witness separation (same estimator both sides)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "separation_curve.png", dpi=130)
    plt.close(fig)
    numbers["separation_rms_um"] = {
        s: {k: float(np.sqrt(np.mean((np.array(v[k]) - np.array(v["target"])) ** 2)))
            for k in ("fit", "null")} for s, v in sep.items()}

    # ---------------- fig 5: off-spot check --------------------------------- #
    off_t = R["tgt"]["off"].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (comp, idx) in zip(axes, (("σ_x", 0), ("σ_y", 1))):
        s_t = sigmas_um(off_t)[idx]
        s_f = sigmas_um(R["off"])[idx]
        s_n = sigmas_um(R["null_off"])[idx]
        ax.plot(scan, s_t, "ko-", ms=3, label="real off (pooled)")
        ax.plot(scan, s_f, "C0o-", ms=3, label="MAP fit (V=0)")
        ax.plot(scan, s_n, "o-", color="0.6", ms=3, label="δ=0 null (V=0)")
        ax.set_xlabel("L2 scan value [deg]")
        ax.set_ylabel(f"{comp} [µm]")
        ax.legend(fontsize=8)
    fig.suptitle("TCAV-off spot (off-channel is IN the loss — residual check)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "offspot.png", dpi=130)
    plt.close(fig)
    numbers["offspot_um"] = {
        "real_sx_median": float(np.median(sigmas_um(off_t)[0])),
        "fit_sx_median": float(np.median(sigmas_um(R["off"])[0])),
        "null_sx_median": float(np.median(sigmas_um(R["null_off"])[0])),
        "real_sy_median": float(np.median(sigmas_um(off_t)[1])),
        "fit_sy_median": float(np.median(sigmas_um(R["off"])[1])),
        "null_sy_median": float(np.median(sigmas_um(R["null_off"])[1])),
    }

    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps({k: v for k, v in numbers.items() if k != "hypotheses"},
                     indent=1))
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
