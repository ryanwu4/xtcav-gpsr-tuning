"""Figures for the Stage-A lab-notebook write-up.

Fig 1: TDC bracket agreement (baseline): DTOTR x-projections Bmad vs Cheetah
       (off + plus) and per-particle residual histogram vs the camera pixel.
Fig 2: The phase-reference finding: measured Bmad kick vs z with the analytic
       slope, and the PHI0 centroid test (Bmad vs Cheetah at -2/0/+2 deg).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning/src")
import torch

from xtcav_gpsr_tuning import paths
from xtcav_gpsr_tuning.convert import C_LIGHT, particlegroup_to_beam
from xtcav_gpsr_tuning.segment import (
    back_drift,
    build_segment_from_spec,
    load_spec,
    set_tdc,
)

paths.ensure_import_roots()
from pmd_beamphysics import ParticleGroup

BLUE = "#0072B2"  # Bmad
VERM = "#D55E00"  # Cheetah
GRAY = "#666666"

IMG = "/home/rwu4/photoinjector-rl/lab-notebook/images"

spec = load_spec("/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning/configs/segment_baseline.json")
E_ref = spec["meta"]["ref_energy_eV"]
L_tdc = spec["meta"]["tdc_length_m"]
d = paths.DATA_DIR / "bmad_reference_baseline"

Px_off = ParticleGroup(str(d / "beam_XTCAVF_off.h5"))
Pd_off = ParticleGroup(str(d / "beam_DTOTR_off.h5"))
Pd_plus = ParticleGroup(str(d / "beam_DTOTR_plus.h5"))
Px_plus = ParticleGroup(str(d / "beam_XTCAVF_plus.h5"))

beam_off, _ = particlegroup_to_beam(Px_off, E_ref)
entrance = back_drift(beam_off, L_tdc)
seg_exit, _ = build_segment_from_spec(spec, start_after="XTCAVF")
seg_full, _ = build_segment_from_spec(spec)
set_tdc(seg_full, 20e6, 0.0)

out_off = seg_exit.track(beam_off)
out_plus = seg_full.track(entrance)

w = Px_off.weight * 1e9  # nC

# ------------------------------------------------------------------ Fig 1 --- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

bins = np.linspace(-3.5, 3.5, 180)
for x, label, color, ls in [
    (np.asarray(Pd_off.x) * 1e3, "Bmad off", BLUE, "-"),
    (out_off.x.detach().numpy() * 1e3, "Cheetah off", VERM, "--"),
    (np.asarray(Pd_plus.x) * 1e3, "Bmad +20 MV", BLUE, "-"),
    (out_plus.x.detach().numpy() * 1e3, "Cheetah +20 MV", VERM, "--"),
]:
    ax1.hist(x, bins=bins, weights=w, histtype="step", lw=1.6, color=color,
             ls=ls, label=label, alpha=0.95)
ax1.set_xlabel("x at DTOTR [mm]")
ax1.set_ylabel("charge [nC / bin]")
ax1.set_title("DTOTR x-projection, exit-plane bracket")
ax1.annotate("off", xy=(0.44, 0.92), xycoords="axes fraction", color=GRAY)
ax1.annotate("+20 MV streak", xy=(0.63, 0.30), xycoords="axes fraction", color=GRAY)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:2], ["Bmad", "Cheetah"], frameon=False, loc="upper right")

dx_um = (out_plus.x.detach().numpy() - np.asarray(Pd_plus.x)) * 1e6
ax2.hist(dx_um, bins=120, weights=w, color=VERM, alpha=0.85)
ax2.axvline(-30.5, color=GRAY, ls=":", lw=1.2)
ax2.axvline(+30.5, color=GRAY, ls=":", lw=1.2)
ax2.annotate("±1 DTOTR2 pixel", xy=(0.62, 0.9), xycoords="axes fraction", color=GRAY)
ax2.set_xlabel("Δx per particle, Cheetah − Bmad [µm]")
ax2.set_ylabel("charge [nC / bin]")
ax2.set_title(f"+20 MV residuals (rms {np.sqrt(np.average(dx_um**2, weights=w)):.2f} µm)")
fig.tight_layout()
fig.savefig(f"{IMG}/2026-07-30_stageA_bracket.png", dpi=160)
plt.close(fig)

# ------------------------------------------------------------------ Fig 2 --- #
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

p0c = np.sqrt(E_ref**2 - 0.51099895069e6**2)
z1_um = -C_LIGHT * (Px_off.t - np.average(Px_off.t, weights=Px_off.weight)) * 1e6
dpx_bmad = (Px_plus.px - Px_off.px) / p0c
sub = np.random.default_rng(0).choice(len(z1_um), 2500, replace=False)
ax1.plot(z1_um[sub], dpx_bmad[sub] * 1e6, ".", ms=2.2, color=BLUE, alpha=0.4,
         label="Bmad, measured per particle")
k_rf = 2 * np.pi * 11.424e9 / C_LIGHT
zz = np.linspace(z1_um.min(), z1_um.max(), 2)
ax1.plot(zz, 20e6 * k_rf / p0c * zz, "-", lw=1.8, color=VERM,
         label=r"analytic $V k_{rf}/p_0 c$ (slope 0.4789 m$^{-1}$)")
ax1.set_xlabel("z at XTCAVF exit [µm]  (head at +z)")
ax1.set_ylabel(r"TDC kick $\Delta p_x/p_0$ [µrad]")
ax1.set_title("Measured Bmad TDC kick at PHI0=0", fontsize=10)
ax1.legend(frameon=False, loc="upper left", fontsize=8)

phis = np.array([-2.0, 0.0, 2.0])
cent_b, cent_c = [], []
for case, phi in [("phi0m", -2.0), ("plus", 0.0), ("phi0p", 2.0)]:
    P = ParticleGroup(str(d / f"beam_DTOTR_{case}.h5"))
    cent_b.append(1e3 * np.average(P.x, weights=P.weight))
    set_tdc(seg_full, 20e6, phi)
    out = seg_full.track(entrance)
    ww = out.particle_charges / out.particle_charges.sum()
    cent_c.append(float((ww * out.x).sum()) * 1e3)
ax2.plot(phis, cent_b, "o", ms=9, color=BLUE, label="Bmad", zorder=3)
ax2.plot(phis, cent_c, "x", ms=9, mew=2.2, color=VERM, label="Cheetah", zorder=4)
for p, b, c in zip(phis, cent_b, cent_c):
    ax2.annotate(f"Δ={abs(b-c)*1e3:.1f} µm", xy=(p, b), xytext=(p + 0.12, b),
                 fontsize=8, color=GRAY)
ax2.axhline(0, color=GRAY, lw=0.8, ls=":")
ax2.set_xlabel("XTCAVF PHI0 [deg]")
ax2.set_ylabel("x centroid at DTOTR [mm]")
ax2.set_title("PHI0 centroid test, Bmad vs Cheetah", fontsize=10)
ax2.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(f"{IMG}/2026-07-30_stageA_phase.png", dpi=160)
plt.close(fig)

print("wrote figures; bracket rms(dx) = %.3f um" % np.sqrt(np.average(dx_um**2, weights=w)))
