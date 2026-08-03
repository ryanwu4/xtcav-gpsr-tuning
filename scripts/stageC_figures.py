"""Figures for the Stage-C lab-notebook write-up.

Fig 1: lambda L-curve (both centering conventions) + projection-based recovery
       (truth vs fitted on the 5 identifiable directions; self- and Bmad-truth).
Fig 2: identifiability spectrum + per-parameter leverage.
Fig 3: image-gap example — Bmad target vs flow render vs |difference|.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

REPO = Path("/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning")
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))

BLUE, VERM, GRAY, GREEN = "#0072B2", "#D55E00", "#666666", "#009E73"
IMG = "/home/rwu4/photoinjector-rl/lab-notebook/images"
SC = REPO / "results" / "stageC"

sr = json.loads((SC / "selfrecovery" / "selfrecovery.json").read_text())
ident = json.loads((SC / "selfrecovery" / "identifiability.json").read_text())
br = json.loads((SC / "bmadrecovery.json").read_text())
cfg = yaml.safe_load((REPO / "configs" / "fit_stageC.yaml").read_text())
sigma_phys = {k: float(v) for k, v in cfg["sigma_phys"].items()}

names = ident["param_names"]
V = np.array(ident["kept_directions"])
knob_names = [n.split(":", 1)[1] for n in names if n.startswith("delta:")]
nk = len(knob_names)


def theta(pulls, nu_y=1.0, dtr=(0, 0, 0), veff_mv=0.0):
    th = np.zeros(len(names))
    for i, k in enumerate(knob_names):
        th[i] = pulls.get(k, 0.0)
    th[nk:nk + 3] = np.array(dtr) / 0.01
    th[nk + 3] = np.log(nu_y)
    th[nk + 4] = veff_mv
    return th


# ------------------------------------------------------------------ Fig 1 --- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))

import collections
for anchor, color, ls in (("com", BLUE, "-"), ("major_lobe", VERM, "--")):
    agg = collections.defaultdict(list)
    for r in sr["lcurve"]:
        if r["anchor"] == anchor:
            agg[r["lambda"]].append(r["pull_err_rms"])
    lams = sorted(agg)
    mean = [np.mean(agg[L]) for L in lams]
    spread = [np.std(agg[L]) for L in lams]
    ax1.errorbar(lams, mean, yerr=spread, color=color, ls=ls, marker="o", ms=5,
                 capsize=3, label=f"anchor = {anchor}")
ax1.axvline(sr["lambda_star"], color=GRAY, ls=":", lw=1.2)
ax1.annotate("λ*", xy=(sr["lambda_star"], ax1.get_ylim()[1] * 0.95), color=GRAY)
ax1.set_xscale("log")
ax1.set_xlabel("λ (MAP prior weight)")
ax1.set_ylabel("pull-error RMS (all 17 knobs)")
ax1.set_title("λ calibration (flow→flow, independent-noise targets)", fontsize=10)
ax1.legend(frameon=False, fontsize=8)

# projections: self-recovery (per scenario, mean over seeds) and Bmad-truth
def proj_points(recs, truth_pulls):
    t = V @ theta(truth_pulls)
    f = np.array([V @ theta(r["pulls"], r["nu_y"], r["dtr"],
                            (r.get("veff", 22e6) - 22e6) / 1e6) for r in recs])
    return t, f.mean(0), f.std(0)

for scen in ("mover", "energy", "quad", "mixed"):
    recs = [r for r in sr["recovery"] if r["scenario"] == scen]
    t, f, s = proj_points(recs, recs[0]["truth_pulls"])
    ax2.errorbar(t, f, yerr=s, fmt="o", ms=6, color=BLUE, alpha=0.85,
                 capsize=2, label="flow→flow" if scen == "mover" else None)
for scen in ("mover", "energy"):
    data = br["scenarios"][scen]
    truth_pulls = {k: dv / sigma_phys[k] for k, dv in data["delta_true_phys"].items()
                   if k in sigma_phys}
    t, f, s = proj_points(data["fits"], truth_pulls)
    ax2.errorbar(t, f, yerr=s, fmt="s", ms=6, color=VERM, alpha=0.9,
                 capsize=2, label="Bmad truth" if scen == "mover" else None)
lim = 1.0
ax2.plot([-lim, lim], [-lim, lim], "-", color=GRAY, lw=0.9)
ax2.set_xlabel("truth projection (prior-scaled)")
ax2.set_ylabel("fitted projection")
ax2.set_title("Recovery on the 5 identifiable directions", fontsize=10)
ax2.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(f"{IMG}/2026-08-03_stageC_recovery.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ Fig 2 --- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                               gridspec_kw={"width_ratios": [1, 1.6]})
S = np.array(ident["singular_values"])
ax1.semilogy(np.arange(1, len(S) + 1), S / S[0], "o-", color=BLUE, ms=4)
ax1.axhline(ident["cut"], color=VERM, ls="--", lw=1.2)
ax1.annotate(f"cut = {ident['cut']} → keep {ident['n_kept']}",
             xy=(0.35, 0.9), xycoords="axes fraction", color=VERM, fontsize=9)
ax1.set_xlabel("singular direction")
ax1.set_ylabel("σ_i / σ_max")
ax1.set_title("Image-Jacobian spectrum (prior-scaled)", fontsize=10)

lev = ident["leverage"]
order = sorted(lev, key=lev.get)
vals = [lev[k] for k in order]
labels = [k.replace("delta:", "") for k in order]
colors = [GREEN if v >= 0.05 else GRAY for v in vals]
ax2.barh(np.arange(len(order)), vals, color=colors, alpha=0.85)
ax2.axvline(0.05, color=VERM, ls="--", lw=1)
ax2.set_yticks(np.arange(len(order)))
ax2.set_yticklabels(labels, fontsize=7)
ax2.set_xlabel("identifiability leverage (norm.)")
ax2.set_xscale("log")
ax2.set_title("Per-parameter leverage over the 7-step ±1° L2 scan", fontsize=10)
fig.tight_layout()
fig.savefig(f"{IMG}/2026-08-03_stageC_identifiability.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ Fig 3 --- #
from xtcav_gpsr_tuning.fit import ForwardModel, load_setup  # noqa: E402
from xtcav_gpsr_tuning.targets import load_dtotr_target  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
fcfg, setup = load_setup(device=device)
valt = paths.TWOBUNCH_ROOT / "data" / "xtcav23_full_valtracks"
man = json.loads((valt / "valtracks_manifest.json").read_text())
tag = sorted(man)[0]
rec = man[tag]
keys = setup.norm["knob_keys"]
lo, hi = np.array(setup.norm["knob_low"]), np.array(setup.norm["knob_high"])
u = (np.array([rec["knobs"][k] for k in keys]) - lo) / (hi - lo)
base = torch.tensor(u, dtype=torch.float64).unsqueeze(0).to(device)
model = ForwardModel(setup, base, rec["voltage_V"], n_particles=int(fcfg["n_particles"]),
                     seed=0).to(device)
with torch.no_grad():
    f_imgs, _ = model()
t_img = load_dtotr_target(valt / f"{tag}_DTOTR_plus.h5", setup.grid).cpu().numpy()
m_img = f_imgs[0, 0].cpu().numpy()
tvgap = np.abs(m_img - t_img).sum()

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.0))
ext = [-4.56, 4.56, -2.42, 2.42]  # mm, analysis frame
vmax = max(t_img.max(), m_img.max())
for ax, im, title in ((axes[0], t_img, f"Bmad target ({tag}, +22 MV, 100k)"),
                      (axes[1], m_img, "flow → segment → screen (δ=0)"),
                      (axes[2], np.abs(m_img - t_img),
                       f"|difference|  (TV = {tvgap:.2f}; a ~p90 case, campaign median 0.77)")):
    h = ax.imshow(im, origin="lower", extent=ext, aspect="auto",
                  cmap="viridis", vmax=vmax if title[0] != "|" else None)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x streak [mm]")
axes[0].set_ylabel("y energy [mm]")
fig.tight_layout()
fig.savefig(f"{IMG}/2026-08-03_stageC_imagegap.png", dpi=160)
plt.close(fig)

print(f"wrote 3 figures; example TV gap {tvgap:.3f}")
