"""Stage-D2: figure for the quad-field-hold probe (stageD2_l2hold_compare.py output).

Per scan step, each geometry (A: LI11 in phase, B: LI11 on crest) with the quad fields held at the
scan centre (solid) or the scan start (dashed), against the same geometry re-LEM'd every step:
image TV with the TCAV on (mean of the two streak signs) and off, and the drive's 90% horizontal
size at XTCAVF relative to the re-LEM'd row.

  python scripts/stageD2_l2hold_figure.py
-> results/stageD2/15673/figures/l2hold.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
GEOM = {"l2m_A": ("A: LI11 in phase", "#2a78d6"), "l2m_B": ("B: LI11 on crest", "#eb6834")}
HOLD = {0.0: ("LEM at scan centre", "-", 4.0), -4.0: ("LEM at scan start", (0, (4, 3)), 0.0)}


def main():
    d = json.loads((REPO / "data/s2e_l2hold_15673/compare_B.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), facecolor=BG)
    for g, (glabel, c) in GEOM.items():
        for h, (hlabel, ls, _) in HOLD.items():
            rows = sorted((o for o in d if o["geometry"] == g and o["hold_delta_deg"] == h),
                          key=lambda o: o["step"])
            x = [0.25 * o["step"] for o in rows]
            kw = dict(color=c, ls=ls, lw=2, marker="o", ms=3.5, label=f"{glabel}, {hlabel}")
            axes[0].plot(x, [(o["vs_relem"]["plus"] + o["vs_relem"]["minus"]) / 2 for o in rows], **kw)
            axes[1].plot(x, [o["vs_relem"]["off"] for o in rows], **kw)
            axes[2].plot(x, [o["held"]["sigma90_drive_x_um"] / o["relem"]["sigma90_drive_x_um"]
                             for o in rows], **kw)
    for ax, title, ylabel in zip(
            axes, ("TCAV on: image change", "TCAV off: image change", "Drive horizontal size at XTCAVF"),
            ("S2E image TV vs re-LEM'd (mean of ±streak)", "S2E image TV vs re-LEM'd (TCAV off)",
             "σx (90%), fields held / re-LEM'd")):
        ax.set_facecolor(BG)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
        ax.set_xlabel("L2 scan value (deg, multiknob)", color=INK2, fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8)
        for _, _, x0 in HOLD.values():
            ax.axvline(x0, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylim(0, 0.6)
    axes[1].set_ylim(0, 0.6)
    axes[2].axhline(1, color=INK2, lw=1)
    axes[0].legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    out = REPO / "results/stageD2/15673/figures/l2hold.png"
    fig.savefig(out, dpi=150, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()
