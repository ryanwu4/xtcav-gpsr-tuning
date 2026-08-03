"""Figures for the xtcav23-surrogate lab-notebook entry.

Fig A: R2 dumbbell — xtcav23 @ XTCAVF vs combined_ft @ PENT, key observables.
Fig B: campaign TPS-emittance coverage vs golden + the real-data-implied band.
Fig C: compact learned knob-response excerpt (movers + kickers, 3 observables).
"""

import json
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src")

BLUE = "#0072B2"   # new (xtcav23)
VERM = "#D55E00"   # old (combined_ft) / reference
GRAY = "#666666"
IMG = "/home/rwu4/photoinjector-rl/lab-notebook/images"
TB = "/home/rwu4/photoinjector-rl/twobunch-s2e-rl"

# ------------------------------------------------------------------ Fig A --- #
new = json.load(open(f"{TB}/results/surrogate/xtcav23/metrics.json"))
old = json.load(open(f"{TB}/results/surrogate/combined_ft/metrics.json"))
keys = [
    ("witness_norm_emit_x", "witness ε_n,x"),
    ("witness_norm_emit_y", "witness ε_n,y"),
    ("energy_difference", "energy difference"),
    ("witness_energy_spread", "witness σ_E"),
    ("transverse_offset", "transverse offset"),
    ("drive_norm_emit_x", "drive ε_n,x"),
    ("drive_sigma_z", "drive σ_z"),
    ("bunch_spacing", "bunch spacing"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ys = np.arange(len(keys))[::-1]
for y, (k, lab) in zip(ys, keys):
    r_old, r_new = old[k]["r2"], new[k]["r2"]
    ax.plot([r_old, r_new], [y, y], "-", color=GRAY, lw=1.2, zorder=1)
    ax.plot(r_old, y, "o", ms=8, color=VERM, zorder=3)
    ax.plot(r_new, y, "o", ms=8, color=BLUE, zorder=3)
    ax.annotate(f"{r_new:.2f}", (r_new, y), xytext=(6, 5),
                textcoords="offset points", fontsize=8, color=BLUE)
ax.set_yticks(ys)
ax.set_yticklabels([lab for _, lab in keys], fontsize=9)
ax.set_xlabel("held-out parity R²")
ax.set_xlim(0.35, 1.02)
ax.axvline(1.0, color=GRAY, lw=0.7, ls=":")
ax.plot([], [], "o", color=VERM, label="combined_ft @ PENT (26-dim, 9008)")
ax.plot([], [], "o", color=BLUE, label="xtcav23 @ XTCAVF (23-dim, 5528)")
ax.legend(frameon=False, loc="lower left", fontsize=8)
ax.set_title("Flow-surrogate parity: XTCAVF handoff vs the PENT predecessor", fontsize=10)
fig.tight_layout()
fig.savefig(f"{IMG}/2026-08-02_xtcav23_parity.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ Fig B --- #
rows, base = [], []
for camp in ("xtcav23_full", "xtcav23_pilot"):
    for fp in sorted(glob(f"{TB}/data/{camp}/sample_*.json")):
        d = json.load(open(fp))
        x = d["specs"].get("XTCAVF", {})
        rec = (x.get("PDrive_norm_emit_x"), x.get("PWitness_norm_emit_x"))
        (base if d.get("is_baseline_repeat") else rows).append(rec)

def clean(vals):
    v = np.array([x for x in vals if x is not None], float)
    return v[np.isfinite(v)] * 1e6

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=False)
for ax, idx, lab in [(axes[0], 0, "drive ε_n,x"), (axes[1], 1, "witness ε_n,x")]:
    v = clean([r[idx] for r in rows])
    g = np.median(clean([b[idx] for b in base]))
    ax.hist(v, bins=np.linspace(0, 70, 90), color=BLUE, alpha=0.85)
    ax.axvline(g, color=GRAY, lw=1.4)
    ax.annotate("golden", (g, ax.get_ylim()[1]), xytext=(2, -12),
                textcoords="offset points", fontsize=8, color=GRAY)
    ax.axvspan(1.5 * g, 3.0 * g, color=VERM, alpha=0.18, lw=0)
    ax.annotate("real-data implied\n(1.5–3× golden)", (2.0 * g, ax.get_ylim()[1] * 0.82),
                fontsize=8, color=VERM, ha="center")
    ax.set_xlabel(f"{lab} at XTCAVF [µm] (90% core)")
    ax.set_ylabel("samples / bin")
    ax.set_title(f"{lab}: golden {g:.1f} µm, campaign median {np.median(v):.1f} µm",
                 fontsize=9)
fig.suptitle("xtcav23 campaign TPS-emittance coverage vs the thickened-line hypothesis",
             fontsize=10, y=1.0)
fig.tight_layout()
fig.savefig(f"{IMG}/2026-08-02_xtcav23_emit_coverage.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ Fig C --- #
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow

CKPT = f"{TB}/trained/xtcav23/checkpoints/best-epoch=498-val_loss=0.1747.ckpt"
model = TwoBunchFlow.load_from_checkpoint(CKPT, map_location="cpu").eval()
norm = json.load(open(f"{TB}/processed/xtcav23_combined_norm.json"))
kk = norm["knob_keys"]
sel = ["S1EL_yOffset", "S2EL_xOffset", "S2ER_xOffset", "S1ELkG", "YC1FFkG", "XC1FFkG"]
obs_defs = [("witness_norm_emit_x", 1e6, "witness ε_n,x [µm]"),
            ("bunch_spacing", 1e6, "spacing [µm]"),
            ("transverse_offset", 1e6, "offset [µm]")]
ngrid = 40
grid = np.linspace(0.02, 0.98, ngrid)
med = np.full(23, 0.5, dtype=np.float32)
K = np.tile(med, (len(sel) * ngrid, 1))
for j, name in enumerate(sel):
    K[j * ngrid:(j + 1) * ngrid, kk.index(name)] = grid
with torch.no_grad():
    torch.manual_seed(0)
    out = model.observables(torch.tensor(K, dtype=torch.float32), n=1024)

fig, ax = plt.subplots(len(obs_defs), len(sel), figsize=(11.5, 5.6),
                       sharex=True)
for oi, (okey, sc, olab) in enumerate(obs_defs):
    y = out[okey].cpu().numpy() * sc
    for j, name in enumerate(sel):
        a = ax[oi, j]
        a.plot(grid, y[j * ngrid:(j + 1) * ngrid], color=BLUE, lw=1.5)
        a.axvline(0.5, color=GRAY, ls=":", lw=0.9)
        a.tick_params(labelsize=7)
        if oi == 0:
            a.set_title(name, fontsize=9)
        if j == 0:
            a.set_ylabel(olab, fontsize=8)
        if oi == len(obs_defs) - 1:
            a.set_xlabel("knob (norm)", fontsize=8)
fig.suptitle("Learned responses on the M2-motivated dims (others at box center; full 23-panel "
             "version in results/surrogate/xtcav23/diagnostics/)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{IMG}/2026-08-02_xtcav23_mover_response.png", dpi=160)
plt.close(fig)

print("wrote 3 figures")
