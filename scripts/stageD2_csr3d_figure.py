"""Stage-D2: does 3D CSR thicken the S2E traces? (probes from stageD2_csr3d_probe.py)

A1_B fitted state, 1e5 particles, 40 CSR bins, convention B, steps 2/10 (v3 rows) and 16/26/29/32
(csrbin32 rows). Variants: 1d-old = the Bmad 20250811 reference probes; on Bmad 20261023: 1d, allb
(steady_state_3d in every BC11/14/20 bend, 1_Dim drifts), all / bc20 (steady_state_3d over whole
chicanes, drift CSR lost), nocsr (CSR off).

Per step (plus sign): TV vs real, drive band width, best per-image blur; per variant: one global blur
over all steps. Beam level at XTCAVF (drive / witness): slice energy spread (1 um z slices, rms about a
linear fit in z, particle-weighted median over slices with >= 30 particles), slice x rms, norm_emit_x.

  PYTHONPATH=$PWD/src python scripts/stageD2_csr3d_figure.py
-> results/stageD2/15673/csr3d.json, results/stageD2/15673/figures/csr3d_plus.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pmd_beamphysics import ParticleGroup  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from stageD2_figures import VMAX_PCT, targets  # noqa: E402
from stageD2_np1e6_figure import best_blur, global_blur  # noqa: E402
from stageD2_thickness import band_width  # noqa: E402
from twobunch_cnf.beam_io import split_bunches  # noqa: E402

VARIANTS = {"1d-old": ("s2e_probe_15673_v3_heater_delay", "s2e_csrbin32_15673"),
            **{v: (f"s2e_csr3d_{v}_lo", f"s2e_csr3d_{v}_hi") for v in ("1d", "allb", "all", "bc20", "nocsr")}}
COLS = {"real": "real", "1d": "S2E, 1D CSR", "allb": "3D CSR in bends", "nocsr": "CSR off"}
STEPS = (2, 10, 16, 26, 29, 32)


def rows(probes):
    """{step: (probe dir, idx)} for the A1B_fit rows with a render."""
    out = {}
    for probe in probes:
        d = REPO / "data" / probe
        for r in json.loads((d / "manifest.json").read_text()):
            if (r["injector_point"] == "A1B_fit" and r["step"] in STEPS and r["step"] not in out
                    and (d / f"render_B_{r['idx']:05d}.npz").exists()):
                out[r["step"]] = (d, r["idx"])
    return out


def slice_stats(B, dz=1e-6, nmin=30):
    z, E, x = np.asarray(B.z), np.asarray(B.energy), np.asarray(B.x)
    k = np.floor((z - z.min()) / dz).astype(int)
    sE, sx, n = [], [], []
    for b in np.unique(k):
        m = k == b
        if m.sum() >= nmin:
            sE.append(np.std(E[m] - np.polyval(np.polyfit(z[m], E[m], 1), z[m])))
            sx.append(np.std(x[m]))
            n.append(m.sum())
    n = np.array(n)

    def wmed(a):
        o = np.argsort(a)
        return float(np.asarray(a)[o][np.searchsorted(np.cumsum(n[o]), n.sum() / 2)])

    return {"slice_dE_MeV": wmed(sE) / 1e6, "slice_x_um": wmed(sx) * 1e6,
            "norm_emit_x_um": float(B.norm_emit_x) * 1e6, "sigma_z_um": float(np.std(z)) * 1e6}


def main():
    real = targets("B")["images"].numpy()
    where = {v: rows(p) for v, p in VARIANTS.items()}
    imgs = {v: {s: np.load(d / f"render_B_{i:05d}.npz") for s, (d, i) in w.items()} for v, w in where.items()}
    steps = [s for s in STEPS if all(s in m for m in imgs.values())]
    res = {"steps": steps, "per_step": {}, "global_blur": {}, "beam": {}}
    for s in steps:
        r = real[0, s]
        rec = {"real_drive_width_um": band_width(r, "drive")}
        for v, m in imgs.items():
            im = m[s]["plus"]
            rec[v] = {"tv_vs_real": float(np.abs(im - r).sum()), "drive_width_um": band_width(im, "drive"),
                      "tv_vs_1d": float(np.abs(im - imgs["1d"][s]["plus"]).sum()), "best_blur": best_blur(im, r)}
        res["per_step"][s] = rec
        print(f"step {s:2d} real w {rec['real_drive_width_um']:.0f} | " + " | ".join(
            f"{v} TV {rec[v]['tv_vs_real']:.3f} w {rec[v]['drive_width_um']:.0f}" for v in imgs), flush=True)
    for v, m in imgs.items():
        res["global_blur"][v] = global_blur({s: m[s] for s in steps}, real, steps)
        d = res["beam"][v] = {}
        for s, (pd, i) in where[v].items():
            D, W = split_bunches(ParticleGroup(str(pd / f"sample_{i:05d}_XTCAVF.h5")))
            d[s] = {"drive": slice_stats(D), "witness": slice_stats(W)}
        print(v, "global blur", res["global_blur"][v], "| drive slice dE MeV",
              {s: round(d[s]["drive"]["slice_dE_MeV"], 2) for s in (2, 29, 32)}, flush=True)
    (REPO / "results/stageD2/15673/csr3d.json").write_text(json.dumps(res, indent=1, default=float))

    fig, axes = plt.subplots(len(steps), len(COLS), figsize=(2.1 * len(COLS), 1.75 * len(steps)), squeeze=False)
    for i, s in enumerate(steps):
        for j, c in enumerate(COLS):
            im = real[0, s] if c == "real" else imgs[c][s]["plus"]
            ax = axes[i, j]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(im, VMAX_PCT))
            ax.set_xticks([]); ax.set_yticks([])
            if c != "real":
                ax.text(0.98, 0.04, f"TV {res['per_step'][s][c]['tv_vs_real']:.2f}", color="w", fontsize=6.5,
                        ha="right", va="bottom", transform=ax.transAxes)
            if i == 0:
                ax.set_title(COLS[c], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"scan {0.25 * s:+.2f}°", fontsize=8)
    fig.suptitle("3D CSR at the A1_B fitted state, DTOTR2 plus, convention B (Bmad 20261023, 1e5)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = REPO / "results/stageD2/15673/figures/csr3d_plus.png"
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
