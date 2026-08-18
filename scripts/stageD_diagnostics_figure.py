"""Stage-D diagnostics figure — the FIT-INDEPENDENT results.

Everything here is either raw data or a δ=0 forward, so none of it depends on
the fitted parameters (and none is affected by the v3 nuisance-prior scaling
bug). Three panels:

  A  off-target cleanup: per-step TCAV-off x-rms before/after the amplitude gate
     + dead-frame floor (audit #2)
  B  orientation contest: δ=0 null image loss in all 4 orientation/pairing
     hypotheses (audit #3 — this is what settles the discrete hypothesis)
  C  handoff back-drift: y-RMS vs real, before/after the 1.0334 m correction,
     with the sdelta ladder (audit #1)

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=$PWD/src:.../twobunch-s2e-rl/src python scripts/stageD_diagnostics_figure.py
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
OLD = REPO / "results" / "stageD" / "15673_v2_buggy_dtr"
PIXEL_UM = 30.5


def off_xrms(npz_path):
    d = np.load(npz_path)
    if "off_xrms_um" in d:
        return d["scan_vals_deg"], d["off_xrms_um"], d["n_off"]
    off = d["off_images"]
    x = np.arange(off.shape[-1]) * PIXEL_UM
    out = []
    for im in off:
        p = im.sum(axis=-2); p = p / p.sum()
        out.append(np.sqrt((p * (x - (p * x).sum()) ** 2).sum()))
    return d["scan_vals_deg"], np.asarray(out), d["n_off"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # ---- A: off-target cleanup ------------------------------------------- #
    ax = axes[0]
    scan, new_x, new_n = off_xrms(D / "targets.npz")
    ax.plot(scan, new_x, "o-", color="C2", ms=4,
            label=f"gated (amp<1, n={new_n.min()}–{new_n.max()}/step)")
    old_npz = OLD / "targets_ungated.npz"
    if old_npz.exists():
        s_o, old_x, old_n = off_xrms(old_npz)
        ax.plot(s_o, old_x, "s--", color="C3", ms=4, alpha=0.8,
                label="ungated (deployed off-cut)")
    ax.axhspan(139, 265, color="0.85", zorder=0,
               label="Tier-0 real off-spot σ range")
    ax.set_yscale("log")
    ax.set_xlabel("L2 scan value [deg]")
    ax.set_ylabel("TCAV-off target x-rms [µm]")
    ax.set_title("A  Off-target contamination removed\n"
                 "(348/400 'off' shots were at 82–98% streak amplitude)",
                 fontsize=9)
    ax.legend(fontsize=7)

    # ---- B: orientation contest ------------------------------------------ #
    ax = axes[1]
    base = json.loads((D / "fit" / "baselines.json").read_text())
    orient = json.loads((D / "fit" / "orient.json").read_text())
    keys = ["flip0_swap0", "flip0_swap1", "flip1_swap0", "flip1_swap1"]
    nulls = [base["null"][k]["img_loss"] for k in keys]
    fits = [orient["runs"][k]["img_loss"] for k in keys]
    xp = np.arange(len(keys))
    ax.bar(xp - 0.2, nulls, 0.4, color="0.6", label="δ=0 null (fit-independent)")
    ax.bar(xp + 0.2, fits, 0.4, color="C0", alpha=0.55,
           label="MAP restart 0")
    floor = json.loads((D / "figures" / "numbers.json").read_text())["floor_tv_median"]
    ax.axhline(floor, color="k", ls="--", lw=1, label="split-half noise floor")
    ax.set_xticks(xp)
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("weighted mean image TV")
    ax.set_title("B  Orientation/pairing hypothesis\n"
                 "flip1_swap1 wins on the NULL alone (1.20 vs 1.47–1.56)",
                 fontsize=9)
    ax.legend(fontsize=7)

    # ---- C: back-drift + sdelta ladder ----------------------------------- #
    ax = axes[2]
    sd = json.loads((D / "sdelta_backdrift_check.json").read_text())
    steps_lbl = ["+0.50°", "+4.00°", "+7.50°"]
    xs = np.arange(3)
    series = [("REAL", "real", "k", "o-"),
              ("no back-drift (v2 bug)", "no_back_drift_old", "C3", "s--"),
              ("back-drift fixed", "back_drift_fixed", "C0", "^-"),
              ("+ σ_δ=1.5", "back_drift_sdelta_1p5", "C2", "v-"),
              ("+ σ_δ=2.0", "back_drift_sdelta_2p0", "C4", "d-")]
    for lbl, key, c, st in series:
        ax.plot(xs, sd[key]["off"], st, color=c, ms=5, label=lbl,
                lw=2 if key == "real" else 1.2)
    ax.set_xticks(xs); ax.set_xticklabels(steps_lbl)
    ax.set_xlabel("scan step")
    ax.set_ylabel("TCAV-off y-rms [µm]")
    ax.set_title("C  Handoff back-drift closes ~41% of the y-deficit\n"
                 "(ratio 2.16 → 1.59; σ_δ carries the modest remainder)",
                 fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Stage-D E300_15673 — fit-independent diagnostics "
                 "(raw data and δ=0 forwards only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "diagnostics.png", dpi=130)
    print(f"figure -> {OUT / 'diagnostics.png'}")

    summary = {
        "off_xrms_gated_um": {"median": float(np.median(new_x)),
                              "min": float(new_x.min()), "max": float(new_x.max())},
        "null_by_orientation": dict(zip(keys, nulls)),
        "orient_restart0_by_orientation": dict(zip(keys, fits)),
        "floor_tv_median": floor,
        "note": ("Panels A and C and the null bars in B are fit-independent (raw "
                 "data / delta=0 forwards). The MAP bars in B are restart 0 of the "
                 "current run's orient stage."),
    }
    (OUT / "diagnostics.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
