"""Stage-D D3.7 figure: is delta-hat a machine property or a scan-region artifact?

Left  — per-knob pulls from the two compression halves against the full-scan fit,
        with the even/odd control band (matched statistics, matched compression)
        drawn as the yardstick for "how much disagreement is just noise".
Right — |low - deep| vs |even - odd| per knob: points far above the diagonal are
        knobs whose fitted value depends on WHICH PART of the scan you fit.
"""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = REPO / "results" / "stageD" / "15673"
c = json.loads((D / "fit" / "consistency.json").read_text())
reg = json.loads((D / "precal" / "register.json").read_text())
e = json.loads((D / "fit" / "ensemble.json").read_text())
h = e["hypotheses"]["flip1_swap1"]
full = h["runs"][h["best_restart"]]["pulls"]
conv = [h["runs"][i] for i in h["converged_indices"]]
lev = {k.split(":", 1)[1]: v for k, v in reg["leverage"].items() if k.startswith("delta:")}
ld, eo = c["delta_low_vs_deep"]["per_knob"], c["delta_even_vs_odd"]["per_knob"]
knobs = sorted(full, key=lambda k: -abs(full[k]))
xp = np.arange(len(knobs))

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0),
                         gridspec_kw={"width_ratios": [2.1, 1]})
ax = axes[0]
ens = [np.ptp([r["pulls"][k] for r in conv]) / 2 for k in knobs]
ax.errorbar(xp, [full[k] for k in knobs], yerr=ens, fmt="ks", ms=6, lw=1.4,
            capsize=3, label="full scan (bars = 8-restart ensemble ±)", zorder=5)
ax.plot(xp, [c["splits"]["low"]["pulls"][k] for k in knobs], "o", color="C0",
        ms=6, label="low compression (steps 0–16)")
ax.plot(xp, [c["splits"]["deep"]["pulls"][k] for k in knobs], "^", color="C3",
        ms=6, label="deep compression (steps 17–32)")
ax.plot(xp, [c["splits"]["even"]["pulls"][k] for k in knobs], "x", color="0.55",
        ms=6, label="even / odd steps (control)")
ax.plot(xp, [c["splits"]["odd"]["pulls"][k] for k in knobs], "+", color="0.55", ms=7)
for y in (-1, 1):
    ax.axhline(y, color="0.75", ls=":", lw=0.8)
ax.axhline(0, color="k", lw=0.6)
for i, k in enumerate(knobs):
    if lev.get(k, 0) < 0.05:
        ax.axvspan(i - 0.5, i + 0.5, color="0.93", zorder=0)
ax.set_xticks(xp); ax.set_xticklabels(knobs, rotation=62, ha="right", fontsize=7)
ax.set_ylabel(r"pull  $\hat\delta_j/\sigma_j$")
ax.set_title("Fitted knob-δ by scan subset — the PATTERN survives (corr +0.91)\n"
             "but MAGNITUDES move ~0.3–0.6σ between compression halves\n"
             "(grey columns = leverage < 0.05, no per-knob claim)", fontsize=9)
ax.legend(fontsize=7, ncol=2)

ax = axes[1]
kept = [k for k in knobs if lev.get(k, 0) >= 0.05]
ax.scatter([abs(eo[k]) for k in kept], [abs(ld[k]) for k in kept], s=46,
           color="C0", label="leverage ≥ 0.05", zorder=4)
ax.scatter([abs(eo[k]) for k in knobs if k not in kept],
           [abs(ld[k]) for k in knobs if k not in kept], s=34, color="0.7",
           label="below cut", zorder=3)
lim = max(max(abs(v) for v in ld.values()), 0.12) * 1.15
ax.plot([0, lim], [0, lim], "k--", lw=1, label="equal (pure statistics)")
for k in kept[:4]:
    ax.annotate(k.replace("Offset", ""), (abs(eo[k]), abs(ld[k])),
                fontsize=6.5, xytext=(4, -2), textcoords="offset points")
ax.set_xlim(0, 0.12); ax.set_ylim(0, lim)
ax.set_xlabel("|even − odd|  (statistical control)")
ax.set_ylabel("|low − deep|  (compression dependence)")
rr = c["delta_low_vs_deep"]["rms_pull_difference"] / c["delta_even_vs_odd"]["rms_pull_difference"]
ax.set_title(f"Compression dependence is {rr:.1f}× the statistical control\n"
             f"→ pre-registered consistency rule FAILS", fontsize=9)
ax.legend(fontsize=7, loc="upper left")

fig.suptitle("Stage-D E300_15673 — is δ̂ a machine property or a scan-region artifact?",
             fontsize=12)
fig.tight_layout()
fig.savefig(D / "figures" / "consistency.png", dpi=130, bbox_inches="tight")
out = {"rms_ratio_low_deep_over_control": rr,
       "rms_low_deep_kept": float(np.sqrt(np.mean([ld[k]**2 for k in kept]))),
       "rms_control_kept": float(np.sqrt(np.mean([eo[k]**2 for k in kept]))),
       "corr_low_deep": c["delta_low_vs_deep"]["corr"],
       "corr_even_odd": c["delta_even_vs_odd"]["corr"],
       "nuisances": {n: {s: c["splits"][s][n] for s in ("low","deep","even","odd")}
                     for n in ("sdelta","ex","nu_y")},
       "verdict": "pre-registered consistency rule FAILS: magnitudes are scan-dependent"}
(D / "figures" / "consistency.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
