"""Stage-D2: what the fitted resolution does to the images. Real DTOTR2 next to (a) true S2E at the
A1_B fitted state (1e5, 40 CSR bins, Bmad 20250811), raw and with its best global Gaussian blur vs
real, and (b) the metrology lambda* CNF fit with the fitted resolution term (A1calgr_B_metro_r0),
rendered with the term switched off and on. Plus streak, convention B, the six steps the S2E probes
share.

Blur axes: image rows = energy (sigma_E), columns = time (sigma_t), 30.5 um per px. The CNF renders
need a GPU once (cached as results/stageD2/15673/render/A1calgr_B_metro_r0{,_noblur}.npz).

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src python scripts/stageD2_blur_figure.py
-> results/stageD2/15673/figures/blur_plus.png, results/stageD2/15673/blur_figure.json
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from stageD2_figures import VMAX_PCT, targets  # noqa: E402
from stageD2_np1e6_figure import global_blur  # noqa: E402
from stageD2_thickness import PX  # noqa: E402

FIT = "A1calgr_B_metro_r0"
S2E = ("s2e_probe_15673_v3_heater_delay", "s2e_csrbin32_15673")
STEPS = (2, 10, 16, 26, 29, 32)
RES = REPO / "results/stageD2/15673"


def s2e_rows():
    out = {}
    for probe in S2E:
        d = REPO / "data" / probe
        for r in json.loads((d / "manifest.json").read_text()):
            f = d / f"render_B_{r['idx']:05d}.npz"
            if r["injector_point"] == "A1B_fit" and r["step"] in STEPS and r["step"] not in out and f.exists():
                out[r["step"]] = np.load(f)
    return out


def cnf_renders():
    """{blur on: imgs (2, B, ny, nx)} of the fit's best restart, rendered once on the GPU."""
    paths = {True: RES / f"render/{FIT}.npz", False: RES / f"render/{FIT}_noblur.npz"}
    if not all(p.exists() for p in paths.values()):
        from run_stageD2_reach import OUT, load_run, make_model
        from stageD2_render import best_restart, render
        d = json.loads((OUT / f"{FIT}.json").read_text())
        r = best_restart(d)
        for blur, p in paths.items():
            m, _, _ = make_model(d["arm"], "cuda", seed=r["seed"])
            load_run(m, r)
            m.fit_blur = blur
            meta = {"arm": FIT, "seed": r["seed"], "final_img_loss": r["final_img_loss"], "fit_blur": blur}
            np.savez_compressed(p, meta=np.array(json.dumps(meta)), **render(m))
    return {b: np.load(p)["imgs"] for b, p in paths.items()}


def main():
    real = targets("B")["images"].numpy()
    s2e = s2e_rows()
    gb = global_blur(s2e, real, STEPS)
    s2e_blur = {s: gaussian_filter(s2e[s]["plus"], (gb["sigma_E_px"], gb["sigma_t_px"]), mode="constant")
                for s in STEPS}
    cnf = cnf_renders()
    fit_px = json.loads((RES / f"reach/{FIT}.json").read_text())["runs"][0]["blur_px"]
    cols = {
        "real": ("real DTOTR2", lambda s: real[0, s]),
        "s2e": ("S2E at the A1_B state", lambda s: s2e[s]["plus"]),
        "s2e_blur": (f"S2E + best blur\n(σ_E {gb['sigma_E_px']} / σ_t {gb['sigma_t_px']} px)",
                     lambda s: s2e_blur[s] / s2e_blur[s].sum()),
        "cnf": ("CNF fit (metrology λ*),\nresolution off", lambda s: cnf[False][0, s]),
        "cnf_blur": (f"CNF fit + fitted resolution\n(σ_E {fit_px['streak_energy']:.2f} / "
                     f"σ_t {fit_px['streak_time']:.2f} px)", lambda s: cnf[True][0, s]),
    }
    res = {"fit": FIT, "fitted_blur_px": fit_px, "s2e_global_blur": gb, "per_step": {},
           "cnf_mean_tv_all_streaked": {k: float(np.abs(cnf[b] - real).sum((-2, -1)).mean())
                                        for k, b in (("off", False), ("on", True))}}
    for s in STEPS:
        res["per_step"][s] = {c: float(np.abs(f(s) - real[0, s]).sum()) for c, (_, f) in list(cols.items())[1:]}
        print(f"step {s:2d} TV vs real | " + " | ".join(f"{c} {v:.3f}" for c, v in res["per_step"][s].items()),
              flush=True)
    print("S2E global blur", gb, "| CNF mean TV (66 streaked)", res["cnf_mean_tv_all_streaked"], flush=True)
    (RES / "blur_figure.json").write_text(json.dumps(res, indent=1, default=float))

    fig, axes = plt.subplots(len(STEPS), len(cols), figsize=(2.2 * len(cols), 1.7 * len(STEPS)), squeeze=False)
    for i, s in enumerate(STEPS):
        for j, (c, (title, f)) in enumerate(cols.items()):
            im = f(s)
            ax = axes[i, j]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(im, VMAX_PCT))
            ax.set_xticks([]); ax.set_yticks([])
            if c != "real":
                ax.text(0.98, 0.04, f"TV {res['per_step'][s][c]:.2f}", color="w", fontsize=6.5, ha="right",
                        va="bottom", transform=ax.transAxes)
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"scan {0.25 * s:+.2f}°", fontsize=8)
    fig.suptitle(f"Fitted resolution vs real DTOTR2, + streak, convention B ({PX} µm/px; image rows = energy, "
                 "columns = time)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = RES / "figures/blur_plus.png"
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
