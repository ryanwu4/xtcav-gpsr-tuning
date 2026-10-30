"""Stage-D2: figure for the machine-like L2 scan probe (stageD2_l2machine_compare.py output).

Left: effective L2 phase of each geometry minus the model scan's phase, per step. Right: S2E image
TV (mean of the two streak signs) of each geometry against the standard model scan of the same
state (solid) and against its longitudinally identical _iso twin (dashed, the energy-profile part).

  python scripts/stageD2_l2machine_figure.py
-> results/stageD2/15673/figures/l2machine.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
GEOM = {"l2m_A": ("LI11 in phase with LI12-14", "#2a78d6"),
        "l2m_B": ("LI11 on crest", "#eb6834")}


def main():
    d = json.loads((REPO / "data/s2e_l2machine_15673/compare_B.json").read_text())
    scan = lambda s: 0.25 * s
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=BG)
    for p, (label, c) in GEOM.items():
        rows = sorted((o for o in d["vs_reference"] if o["point"] == p), key=lambda o: o["step"])
        x = [scan(o["step"]) for o in rows]
        a0.plot(x, [o["phi_eff_deg"] - (-37.9 + scan(o["step"])) for o in rows], color=c, lw=2,
                marker="o", ms=4, label=label)
        a1.plot(x, [(o["plus"] + o["minus"]) / 2 for o in rows], color=c, lw=2, marker="o", ms=4,
                label=f"{label}: vs model scan")
        tw = sorted((t for t in d["vs_iso_twin"] if t["point"] == p), key=lambda t: t["step"])
        a1.plot([scan(t["step"]) for t in tw], [(t["plus"] + t["minus"]) / 2 for t in tw], color=c,
                lw=1.5, ls=(0, (4, 3)), label=f"{label}: vs iso twin")
    a0.axhline(0, color=INK2, lw=1)
    a0.set_ylabel("effective L2 phase − model step (deg)", color=INK2, fontsize=9)
    a1.set_ylabel("S2E image TV (mean of ±streak)", color=INK2, fontsize=9)
    a1.set_ylim(0, 1.6)
    for ax in (a0, a1):
        ax.set_facecolor(BG)
        ax.set_xlabel("L2 scan value (deg, multiknob)", color=INK2, fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="upper center",
                  ncol=1 if ax is a0 else 2)
    a0.set_title("Machine-like L2 scan: phase map", fontsize=10, color=INK, loc="left")
    a1.set_title("Image change vs the model's scan", fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out = REPO / "results/stageD2/15673/figures/l2machine.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()
