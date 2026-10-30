"""Stage-D2 arm comparison: weighted image TV of every fit arm (best restart) with its δ = 0
null, the split-half noise floor, and Stage D's published fit. Rows whose result json is
missing are skipped (the wide-model rows appear once those fits exist).

  PYTHONPATH=$PWD/src python scripts/stageD2_arms_figure.py
-> results/stageD2/15673/figures/arms.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stageD2_render import best_restart  # noqa: E402

REACH = REPO / "results/stageD2/15673/reach"
FLOOR = 0.0995          # mean(w * split-half floor), convention B targets
STAGE_D = (1.2024, 0.9581)   # published Stage D null -> fit (xtcav23, convention D, λ* 0.01)
COLORS = {"xtcav23": "#2a78d6", "cnf": "#eb6834", "cnf_wide": "#1baf7a"}
INK, INK2, NULL = "#0b0b0b", "#52514e", "#8a8984"
ROWS = [  # (label, result json stem, model); section headers have stem None
    ("Stage D (published, conv. D, λ*)", "STAGE_D", "xtcav23"),
    ("xtcav23, 17 linac/FF", "X1_B", "xtcav23"),
    ("xtcav23, + dedicated V_eff", "X1cal_B", "xtcav23"),
    ("CNF, 17 linac/FF (injector golden)", "A4_B", "cnf"),
    ("CNF, + drive fraction", "A7_B", "cnf"),
    ("CNF, + 6 active injector", "A5_B", "cnf"),
    ("CNF, + L0B", "A6_B", "cnf"),
    ("CNF, all 31 knobs", "A1_B", "cnf"),
    ("CNF, all 31 + shared latent", "A2_B", "cnf"),
    ("CNF, all 31 + V_eff + L2 gain", "A1calg_B", "cnf"),
    ("wide CNF, all 31 + V_eff + L2 gain", "W1calg_B", "cnf_wide"),
    ("CNF, all 31, λ*", "A1_B_metro_r0", "cnf"),
    ("CNF, all 31 + V_eff + L2 gain, λ*", "A1calg_B_metro_r0", "cnf"),
    ("wide CNF, all 31 + V_eff + L2 gain, λ*", "W1calg_B_metro_r0", "cnf_wide"),
]


def main():
    rows = []
    for label, stem, model in ROWS:
        if stem == "STAGE_D":
            rows.append((label, *STAGE_D, model))
            continue
        f = REACH / f"{stem}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        r = best_restart(d)
        null = next((x["null"]["img"] for x in d["runs"] if x.get("null")), None)
        rows.append((label, null, r["final_img_loss"], model))
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(rows) + 1.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = list(range(len(rows)))[::-1]
    for yi, (label, null, fit, model) in zip(y, rows):
        ax.barh(yi, fit, height=0.62, color=COLORS[model], edgecolor="#fcfcfb", linewidth=2)
        ax.text(fit + 0.012, yi, f"{fit:.3f}", va="center", ha="left", fontsize=8.5, color=INK)
        if null is not None:
            ax.plot([null, null], [yi - 0.34, yi + 0.34], color=NULL, lw=2, solid_capstyle="butt")
    ax.axvline(FLOOR, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(FLOOR + 0.01, y[0] + 0.62, "noise floor", fontsize=8, color=INK2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.32)
    ax.set_xlabel("weighted mean image TV vs real (66 images; lower is better)", fontsize=9, color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS["xtcav23"], label="xtcav23 flow"),
               Patch(color=COLORS["cnf"], label="C_B3_ema CNF")]
    if any(r[3] == "cnf_wide" for r in rows):
        handles.append(Patch(color=COLORS["cnf_wide"], label="wide-box CNF"))
    handles.append(Line2D([], [], color=NULL, lw=2, label="δ = 0 null"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(handles),
              fontsize=8, frameon=False, labelcolor=INK)
    ax.set_title("E300_15673 fit arms (convention B, λ_reach unless marked λ*)", fontsize=10,
                 color=INK, loc="left", pad=22)
    fig.tight_layout()
    out = REPO / "results/stageD2/15673/figures/arms.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()
