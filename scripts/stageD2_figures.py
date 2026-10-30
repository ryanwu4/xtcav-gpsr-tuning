"""Stage-D2 figures from the rendered arms (stageD2_render.py) — CPU only.

  gallery_plus.png / gallery_minus.png  rows = real + models/arms, 6 scan steps
  residual_plus.png / residual_minus.png  model − real, shared symmetric scale per column
  per_step_tv.png   per-image TV vs scan step per arm, split-half floor
  features.png      controller features (spacing, drive σz, drive charge fraction)
                    real vs arms, same estimator on both
  walls.png         final normalized knob position of every fitted knob, per arm
  summary.json      headline numbers per arm (weighted image loss, per-regime TV,
                    feature errors, walls)

  PYTHONPATH=$PWD/src python scripts/stageD2_figures.py --rows null_xtcav23 X1_xtcav23 null_cb3 A1_all_mean ...
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from run_stageD2_reach import ARMS, OUT, TARGET_FILES  # noqa: E402
from run_stageD_fit import orient_targets  # noqa: E402
from xtcav_gpsr_tuning.features_score import FEATURES, feature_errors, scan_features  # noqa: E402
from xtcav_gpsr_tuning.screen import PIXEL_UM  # noqa: E402
from xtcav_gpsr_tuning.stage_d import load_targets  # noqa: E402

RENDER = OUT.parent / "render"
FIG = OUT.parent / "figures"
STEPS = [2, 8, 16, 20, 26, 30]        # scan +0.5, 2, 4, 5, 6.5, 7.5 deg (incl. Stage D's 3 panels)
VMAX_PCT = 99.5
_BASE_LABELS = {"A1": "C_B3 knobs+inj", "A2": "+ shared latent", "A3": "+ per-step latent",
                "A4": "C_B3 linac only", "A5": "+ active 6 inj", "A6": "+ active 6 + L0B",
                "X1": "xtcav23 fit (D2)"}
_CONV = {"D": "[D]", "N": "[N]", "B": "[B]"}
LABELS = {"real": "real target",
          **{f"{a}_{c}": f"{lab} {_CONV[c]}" for a, lab in _BASE_LABELS.items() for c in _CONV},
          **{f"null_{m}_{c}": f"{name} δ=0 {_CONV[c]}" for m, name in (("cb3", "C_B3"),
                                                                      ("xtcav23", "xtcav23"))
             for c in _CONV}}
STREAK = -14.402902846219119   # features_score.streak_dxdz at the anchored V_eff (checked 2026-10-29)


def targets(variant):
    variant = {"D": "deployed", "N": "none", "B": "clean"}.get(variant, variant)
    cfg = {"targets_npz": TARGET_FILES[variant] or "results/stageD/15673/targets.npz"}
    return orient_targets(load_targets(cfg), True, True)


def load_rows(names):
    rows = {}
    for n in names:
        d = np.load(RENDER / f"{n}.npz")
        rows[n] = {"imgs": d["imgs"], "off": d["off"], "meta": json.loads(str(d["meta"]))}
    return rows


def regime_tv(tv, scan):
    out = {}
    for name, (lo, hi) in {"low_0_3": (0, 3), "mid_3_6": (3, 6), "deep_6_8": (6, 8.01)}.items():
        sel = (scan >= lo) & (scan < hi)
        out[name] = float(tv[:, sel].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--variant", default="D", help="row-correction convention of the real row "
                    "(D / N / B); every row in one figure must share it")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    real = targets(args.variant)
    rimg = real["images"].numpy()
    scan = np.asarray(load_targets({"targets_npz": "results/stageD/15673/targets.npz"})
                      ["scan_vals_deg"], float)
    floors = real["floors"].numpy()
    w = real["weights"].numpy()
    rows = load_rows(args.rows)
    tag = args.tag

    # ---- galleries + residuals ----
    for gi, sign in enumerate(("plus", "minus")):
        names = ["real"] + args.rows
        fig, axes = plt.subplots(len(names), len(STEPS), figsize=(2.6 * len(STEPS), 1.55 * len(names)),
                                 squeeze=False)
        for c, si in enumerate(STEPS):
            for r, n in enumerate(names):
                im = rimg[gi, si] if n == "real" else rows[n]["imgs"][gi, si]
                ax = axes[r, c]
                ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0,
                          vmax=np.percentile(im, VMAX_PCT))
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"scan {scan[si]:+.2f}°", fontsize=9)
                if c == 0:
                    ax.set_ylabel(LABELS.get(n, n), fontsize=7.5)
                if n != "real":
                    tv = np.abs(im - rimg[gi, si]).sum()
                    ax.text(0.98, 0.04, f"TV {tv:.2f}", color="w", fontsize=7, ha="right",
                            va="bottom", transform=ax.transAxes)
        fig.suptitle(f"E300_15673 {('+', '−')[gi]} streak sign: real vs models "
                     f"(each panel scaled to its own {VMAX_PCT}th pct; TV vs the real panel)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"gallery_{sign}{tag}.png", dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(len(args.rows), len(STEPS),
                                 figsize=(2.6 * len(STEPS), 1.55 * len(args.rows)), squeeze=False)
        for c, si in enumerate(STEPS):
            lim = max(np.abs(rows[n]["imgs"][gi, si] - rimg[gi, si]).max() for n in args.rows)
            for r, n in enumerate(args.rows):
                ax = axes[r, c]
                ax.imshow(rows[n]["imgs"][gi, si] - rimg[gi, si], origin="lower", aspect="auto",
                          cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"scan {scan[si]:+.2f}°", fontsize=9)
                if c == 0:
                    ax.set_ylabel(LABELS.get(n, n), fontsize=7.5)
        fig.suptitle(f"model − real ({('+', '−')[gi]} sign): red = model has excess intensity, "
                     "blue = model is missing it; symmetric scale shared down each column", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"residual_{sign}{tag}.png", dpi=120)
        plt.close(fig)

    # ---- per-step TV ----
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    cmap = plt.get_cmap("tab10")
    for i, n in enumerate(args.rows):
        tv = np.abs(rows[n]["imgs"] - rimg).sum(axis=(-2, -1))   # (2, B)
        summary[n] = {"img_loss_weighted": float((tv * w).mean()), "tv_unweighted_mean": float(tv.mean()),
                      "regimes": regime_tv(tv, scan),
                      "tv_over_floor_mean": float(np.nanmean(tv / floors)),
                      "off_tv_mean": float(np.abs(rows[n]["off"] - real["off"].numpy()).sum(axis=(-2, -1)).mean())}
        for gi in range(2):
            axes[gi].plot(scan, tv[gi], "o-", ms=2.5, lw=1, color=cmap(i % 10), label=LABELS.get(n, n))
    for gi in range(2):
        axes[gi].plot(scan, floors[gi], "k--", lw=1, label="split-half floor")
        axes[gi].set_xlabel("L2 scan value [deg]")
        axes[gi].set_title(f"streak sign {('+', '−')[gi]}")
    axes[0].set_ylabel("per-image TV vs real (unit-sum)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"per_step_tv{tag}.png", dpi=130)
    plt.close(fig)

    # ---- physics features (same estimator) ----
    fr = scan_features(rimg, STREAK)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    units = {"bunch_spacing": (1e6, "spacing [µm]"), "drive_sigma_z": (1e6, "drive σz [µm]"),
             "drive_charge_fraction": (1.0, "drive charge fraction")}
    for j, k in enumerate(FEATURES):
        sc, lab = units[k]
        axes[j].plot(scan, fr[k] * sc, "ko-", ms=3, lw=1.5, label="real")
        axes[j].set_ylabel(lab); axes[j].set_xlabel("L2 scan value [deg]")
    for i, n in enumerate(args.rows):
        fm = scan_features(rows[n]["imgs"], STREAK)
        summary[n]["features"] = feature_errors(fm, fr)
        for j, k in enumerate(FEATURES):
            sc, _ = units[k]
            axes[j].plot(scan, fm[k] * sc, "o-", ms=2, lw=1, color=cmap(i % 10), label=LABELS.get(n, n))
    axes[0].legend(fontsize=7)
    fig.suptitle("Controller XTCAV features (valley-rule energy cut), same estimator on real and model "
                 "images; gaps = no valley found", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"features{tag}.png", dpi=130)
    plt.close(fig)

    # ---- walls ----
    fits = [n for n in args.rows if (OUT / f"{n}.json").exists()]
    if fits:
        allk = []
        for n in fits:
            d = json.loads((OUT / f"{n}.json").read_text())
            best = min(d["runs"], key=lambda r: r["total_loss"])
            summary[n]["walls"] = best["walls"]
            summary[n]["u_final"] = best["u_final"]
            summary[n]["l1_shift_deg"] = best["l1_shift_deg"]
            summary[n]["l2_shift_deg"] = best["l2_shift_deg"]
            summary[n]["nu_y"] = best["nu_y"]
            summary[n]["dtr"] = best["dtr"]
            summary[n]["latent"] = best.get("latent", {})
            allk += [k for k in best["u_final"] if k not in allk]
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(allk)), 4.5))
        for i, n in enumerate(fits):
            u = summary[n]["u_final"]
            xs = [allk.index(k) + (i - len(fits) / 2) * 0.08 for k in u]
            ax.plot(xs, [0.5 * (a + b) for a, b in u.values()], "o", ms=4, color=cmap(i % 10),
                    label=LABELS.get(n, n))
        ax.axhline(0, color="r", lw=0.8); ax.axhline(1, color="r", lw=0.8)
        ax.set_xticks(range(len(allk)))
        ax.set_xticklabels([k.split(":")[-2] if k.count(":") >= 2 else k for k in allk],
                           rotation=70, fontsize=7)
        ax.set_ylabel("fitted value, normalized to the training box [0, 1]")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f"walls{tag}.png", dpi=130)
        plt.close(fig)
    (FIG / f"summary{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({n: {k: v for k, v in s.items() if k in ("img_loss_weighted", "regimes", "walls")}
                      for n, s in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
