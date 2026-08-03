"""Extended image-gap figures (Ryan's request: contextualize the flow-vs-Bmad
visual gap against parameter-change signal sizes).

Fig A: gallery across valtrack samples at best / median / worst TV gap —
       Bmad target | flow render | |difference|, per row.
Fig B: parameter-sensitivity gallery at the median-gap sample: |Δimage| for
       single-parameter changes at fit-relevant amplitudes, all on ONE color
       scale, with the flow-vs-Bmad gap as the last panel for direct comparison.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path("/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning")
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from xtcav_gpsr_tuning.fit import ForwardModel, load_setup  # noqa: E402
from xtcav_gpsr_tuning.targets import load_dtotr_target  # noqa: E402

IMG = "/home/rwu4/photoinjector-rl/lab-notebook/images"
SC = REPO / "results" / "stageC"
VALT = paths.TWOBUNCH_ROOT / "data" / "xtcav23_full_valtracks"

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg, setup = load_setup(device=device)
man = json.loads((VALT / "valtracks_manifest.json").read_text())
gap = json.loads((SC / "imagegap.json").read_text())
keys = setup.norm["knob_keys"]
lo, hi = np.array(setup.norm["knob_low"]), np.array(setup.norm["knob_high"])
EXT = [-4.56, 4.56, -2.42, 2.42]
N_P = int(cfg["n_particles"])


def flow_image(tag, **param_changes):
    rec = man[tag]
    u = (np.array([rec["knobs"][k] for k in keys]) - lo) / (hi - lo)
    base = torch.tensor(u, dtype=torch.float64).unsqueeze(0).to(device)
    m = ForwardModel(setup, base, rec["voltage_V"], n_particles=N_P, seed=0).to(device)
    with torch.no_grad():
        for name, val in param_changes.items():
            if name == "nu_y":
                m.log_nu_y.copy_(torch.tensor(np.log(val)))
            elif name == "veff_frac":
                m.veff_frac.copy_(torch.tensor(float(val)))
            elif name.startswith("dtr"):
                m.dtr[int(name[3])] = float(val)
            else:  # knob delta in sigma_j units
                j = setup.fit_knobs.index(name)
                sn = float(setup.sigma_norm[j])
                m.delta_raw[j] = torch.atanh(torch.tensor(
                    val * sn / float(setup.bound_norm[j])).clamp(-0.99, 0.99))
        imgs, _ = m()
    return imgs[0, 0].cpu().numpy()  # plus sign


# ------------------------------------------------------------------ Fig A --- #
rows = sorted(gap["rows"], key=lambda r: r["plus_gap_tv"])
picks = [rows[0], rows[len(rows) // 2], rows[-1]]
labels = ["best", "median", "worst"]

fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.2))
for i, (row, lab) in enumerate(zip(picks, labels)):
    tag = row["tag"]
    t_img = load_dtotr_target(VALT / f"{tag}_DTOTR_plus.h5", setup.grid).cpu().numpy()
    m_img = flow_image(tag)
    vmax = max(t_img.max(), m_img.max())
    for j, (im, title, vm) in enumerate((
            (t_img, f"Bmad target — {tag}", vmax),
            (m_img, "flow render (δ=0)", vmax),
            (np.abs(m_img - t_img), f"|diff| TV={row['plus_gap_tv']:.2f} ({lab})", None))):
        ax = axes[i, j]
        ax.imshow(im, origin="lower", extent=EXT, aspect="auto", cmap="viridis", vmax=vm)
        ax.set_title(title, fontsize=9)
        if i == 2:
            ax.set_xlabel("x streak [mm]", fontsize=8)
        if j == 0:
            ax.set_ylabel("y energy [mm]", fontsize=8)
fig.suptitle("Flow-vs-Bmad image gap across the valtrack distribution "
             f"(24 samples: TV median {gap['summary']['gap_tv_median']:.2f}, "
             f"p90 {gap['summary']['gap_tv_p90']:.2f}; sampling floors ~0.06-0.09)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(f"{IMG}/2026-08-03_stageC_gap_gallery.png", dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ Fig B --- #
tag = picks[1]["tag"]  # median-gap sample
base_img = flow_image(tag)
t_img = load_dtotr_target(VALT / f"{tag}_DTOTR_plus.h5", setup.grid).cpu().numpy()

changes = [
    ("S2ER_xOffset", 2.0, "S2ER_x +2σ (200 µm)"),
    ("S1EL_yOffset", 2.0, "S1EL_y +2σ (200 µm)"),
    ("L3EnergyOffset", 1.5, "L3E +1.5σ (16.5 MeV)"),
    ("S2ELkG", 2.0, "S2ELkG +2σ (1%)"),
    ("dtr2", -0.068, "Q0FF k1 −6.8% (the veff absorber)"),
    ("veff_frac", 0.10, "V_eff +10%"),
    ("nu_y", 1.10, "ν_y ×1.10"),
]
diffs, titles = [], []
for name, val, lab in changes:
    kw = {name: val} if not name.startswith(("dtr", "veff", "nu")) else \
        ({"dtr2": val} if name == "dtr2" else {name: val})
    im = flow_image(tag, **kw)
    d = np.abs(im - base_img)
    diffs.append(d)
    titles.append(f"{lab}\nTV={d.sum():.2f}")
gap_d = np.abs(base_img - t_img)

vmax = max(d.max() for d in diffs + [gap_d])
fig, axes = plt.subplots(2, 4, figsize=(12.5, 5.6))
panels = list(zip(diffs, titles)) + [(gap_d,
                                      f"flow-vs-Bmad GAP (δ=0)\nTV={gap_d.sum():.2f}")]
for ax, (im, title) in zip(axes.ravel(), panels):
    ax.imshow(im, origin="lower", extent=EXT, aspect="auto", cmap="magma", vmax=vmax)
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=7)
fig.suptitle(f"|Δimage| from single-parameter changes vs the surrogate gap — {tag}, "
             "+22 MV, SHARED color scale", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{IMG}/2026-08-03_stageC_param_sensitivity.png", dpi=150)
plt.close(fig)

print("param-change TVs:", {t.splitlines()[0]: round(float(d.sum()), 3)
                            for d, t in zip(diffs, titles)})
print("surrogate gap TV (median sample):", round(float(gap_d.sum()), 3))
