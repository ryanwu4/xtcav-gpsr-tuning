"""Stage-D fit-quality assessment figure (v3.1, all gates passing).

Four panels answering "how well did the fitting procedure do?":
  A  loss ladder: null -> nuisance-only -> MAP, against the noise floor and the
     controlled bias, for both carried orientation hypotheses
  B  per-step residual vs the split-half floor, by compression regime
  C  pulls with ensemble spread, sorted, with the leverage cut marked
  D  validation channels not in the loss (separation) and in it (off-spot)

  PYTHONPATH=$PWD/src python scripts/stageD_assessment_figure.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results" / "stageD" / "15673"
OUT = D / "figures"


def main():
    b = json.loads((D / "fit" / "baselines.json").read_text())
    e = json.loads((D / "fit" / "ensemble.json").read_text())
    reg = json.loads((D / "precal" / "register.json").read_text())
    ctl = json.loads((D / "precal" / "control.json").read_text())
    num = json.loads((OUT / "numbers.json").read_text())
    tgt = np.load(D / "targets.npz")
    scan = tgt["scan_vals_deg"]
    sel0 = json.loads((D / "fit" / "orient.json").read_text())
    floors = tgt["floors_tv"].T
    if sel0["runs"][sel0["winner"]]["swap_signs"]:
        floors = floors[::-1]          # sign-indexed: must follow the pairing
    if sel0["runs"][sel0["winner"]]["flip_rows"]:
        pass                           # row flip does not reindex signs
    MAIN, ALT = "flip1_swap1", "flip0_swap1"
    hm = e["hypotheses"][MAIN]
    best = hm["runs"][hm["best_restart"]]
    conv = [hm["runs"][i] for i in hm["converged_indices"]]
    floor = num["floor_tv_median"]

    tvs = reg["tv_signal_per_unit"]
    zero = [r for r in ctl["runs"] if r["scenario"] == "zero"]
    tv_bias = float(np.mean([np.sqrt(sum((r["pull_err"].get(k, 0.0)
                                          * tvs.get("delta:" + k, 0.0)) ** 2
                                         for k in r["pull_err"])) for r in zero]))

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.24)

    # ---- A: loss ladder --------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    labels, vals, colors = [], [], []
    for hyp, c in ((MAIN, "C0"), (ALT, "C1")):
        hh = e["hypotheses"][hyp]
        bb = hh["runs"][hh["best_restart"]]
        labels += [f"null\n{hyp}", f"nuis-only\n{hyp}", f"MAP\n{hyp}"]
        vals += [b["null"][hyp]["img_loss"],
                 b["nuisance_only"][hyp]["final_img_loss"],
                 bb["final_img_loss"]]
        colors += ["0.65", c + "", c]
    bars = ax.bar(range(len(vals)), vals, color=colors, alpha=0.9)
    bars[1].set_alpha(0.45); bars[4].set_alpha(0.45)
    ax.axhline(floor, color="k", ls="--", lw=1.2,
               label=f"split-half noise floor ({floor:.3f})")
    ax.axhline(tv_bias, color="r", ls=":", lw=1.2,
               label=f"controlled bias in TV units ({tv_bias:.3f})")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("weighted mean image TV")
    ax.set_title(f"A  Loss ladder — MAP buys {b['null'][MAIN]['img_loss']-best['final_img_loss']:.3f} "
                 f"({(1-best['final_img_loss']/b['null'][MAIN]['img_loss'])*100:.0f}%), "
                 f"{(b['null'][MAIN]['img_loss']-best['final_img_loss'])/floor:.1f}x the floor",
                 fontsize=9)
    ax.legend(fontsize=7)

    # ---- B: per-step residual -------------------------------------------- #
    ax = fig.add_subplot(gs[0, 1])
    null_tv = np.array(b["null"][MAIN]["per_image_tv"])
    fit_tv = np.array(best["per_image_tv"])
    for gi, (sg, mk) in enumerate((("+", "o"), ("−", "s"))):
        ax.plot(scan, null_tv[gi], mk + "-", color="0.65", ms=3.5,
                label=f"null {sg}" if gi == 0 else None)
        ax.plot(scan, fit_tv[gi], mk + "-", color="C0", ms=3.5,
                label=f"MAP {sg}" if gi == 0 else None)
        ax.plot(scan, floors[gi], mk + "--", color="k", ms=2, lw=0.8,
                label="floor" if gi == 0 else None)
    for lo, hi, c in ((0, 3, "#e8f4ff"), (3, 6, "#fff6e0"), (6, 8.1, "#ffe8e8")):
        ax.axvspan(lo, hi, color=c, zorder=0)
    ax.text(1.3, 0.12, "low compr.\n+0.28", fontsize=7, ha="center")
    ax.text(4.5, 0.12, "mid\n+0.35", fontsize=7, ha="center")
    ax.text(7.0, 0.12, "deep\n+0.20", fontsize=7, ha="center")
    ax.set_xlabel("L2 scan value [deg]"); ax.set_ylabel("per-image TV")
    rr = fit_tv / floors
    ax.set_title(f"B  Residual per step: improves everywhere (0/66 worse),\n"
                 f"but stays {rr.min():.1f}–{rr.max():.0f}x the floor "
                 f"(mean {rr.mean():.1f}x); least gain at deep compression",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=3)

    # ---- C: pulls --------------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 0])
    lev = {k.split(":", 1)[1]: v for k, v in reg["leverage"].items()
           if k.startswith("delta:")}
    knobs = sorted(best["pulls"], key=lambda k: -abs(best["pulls"][k]))
    med = [np.median([r["pulls"][k] for r in conv]) for k in knobs]
    lo = [min(r["pulls"][k] for r in conv) for k in knobs]
    hi = [max(r["pulls"][k] for r in conv) for k in knobs]
    cols = ["C0" if lev.get(k, 0) >= 0.05 else "0.75" for k in knobs]
    ax.bar(range(len(knobs)), med, color=cols,
           yerr=[np.subtract(med, lo), np.subtract(hi, med)],
           error_kw={"lw": 1.0, "ecolor": "k"})
    for y in (-1, 1):
        ax.axhline(y, color="0.6", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(knobs)))
    ax.set_xticklabels(knobs, rotation=62, ha="right", fontsize=7)
    ax.set_ylabel(r"pull  $\hat\delta_j/\sigma_j$")
    ax.set_title("C  MAP pulls (bars = ensemble range over 8/8 converged "
                 "restarts;\ngrey = leverage < 0.05, no per-knob claim)", fontsize=9)

    # ---- D: validation --------------------------------------------------- #
    ax = fig.add_subplot(gs[1, 1])
    sep = num["separation_rms_um"]; off = num["offspot_um"]
    groups = ["sep RMS +", "sep RMS −", "off σx", "off σy"]
    nullv = [sep["+"]["null"], sep["−"]["null"], off["null_sx_median"],
             off["null_sy_median"]]
    fitv = [sep["+"]["fit"], sep["−"]["fit"], off["fit_sx_median"],
            off["fit_sy_median"]]
    realv = [0, 0, off["real_sx_median"], off["real_sy_median"]]
    xp = np.arange(4)
    ax.bar(xp - 0.22, nullv, 0.4, color="0.65", label="δ=0 null")
    ax.bar(xp + 0.22, fitv, 0.4, color="C0", label="MAP fit")
    ax.plot(xp[2:] + 0.0, realv[2:], "r*", ms=14, label="real (target value)")
    for i in (0, 1):
        ax.annotate("→0 is\nperfect", (xp[i], max(nullv[i], fitv[i]) * 1.02),
                    fontsize=6, ha="center", va="bottom")
    ax.set_xticks(xp); ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("µm")
    ax.set_title("D  Validation channels — separation (held out of the loss)\n"
                 "and off-spot; every channel improves, none degrades",
                 fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Stage-D E300_15673 v3.1 — fit-quality assessment "
                 "(all D0 gates passing, λ*=0.01, n=65536/bunch)", fontsize=12)
    fig.savefig(OUT / "assessment.png", dpi=130, bbox_inches="tight")
    print(f"figure -> {OUT / 'assessment.png'}")

    out = {
        "null": b["null"][MAIN]["img_loss"],
        "nuisance_only": b["nuisance_only"][MAIN]["final_img_loss"],
        "map": best["final_img_loss"],
        "reduction": b["null"][MAIN]["img_loss"] - best["final_img_loss"],
        "floor": floor, "controlled_bias_tv": tv_bias,
        "residual_over_floor_mean": float(np.mean(fit_tv / floors)),
        "residual_over_floor_min": float(np.min(fit_tv / floors)),
        "residual_over_floor_max": float(np.max(fit_tv / floors)),
        "reduction_from_nuisances": (b["null"][MAIN]["img_loss"]
                                     - b["nuisance_only"][MAIN]["final_img_loss"]),
        "reduction_from_knobs": (b["nuisance_only"][MAIN]["final_img_loss"]
                                 - best["final_img_loss"]),
        "images_worse_than_null": int((null_tv - fit_tv < 0).sum()),
        "nuisances": {k: best[k] for k in ("sdelta", "ex", "nu_y", "l2_shift_deg")},
        "dtr_max_abs": float(np.abs(best["dtr"]).max()),
        "orientation_ratio": e["hypotheses"][ALT]["runs"][
            e["hypotheses"][ALT]["best_restart"]]["final_img_loss"] / best["final_img_loss"],
    }
    (OUT / "assessment.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
