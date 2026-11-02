"""Stage-D2 figures for the final notebook entry: plain labels, three scan steps (+0.5 / +4 / +6.5 deg),
+ streak, convention B unless noted. CPU only: reads the cached renders (results/stageD2/15673/render),
the S2E probe renders (data/s2e_*) and the fit records (reach/, metrology_*.json, register_*.json).

  PYTHONPATH=$PWD/src python scripts/stageD2_notebook_figures.py
-> results/stageD2/15673/figures/notebook/{pipeline,chain,gallery,widebox,rowcorr,gain,thickness,
   metrology,ruledout}.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from stageD2_figures import VMAX_PCT, targets  # noqa: E402

RES = REPO / "results/stageD2/15673"
OUT = RES / "figures/notebook"
STEPS = (2, 16, 26)                      # scan +0.5, +4, +6.5 deg
DEEP = 26
BLUE, ORANGE, GRAY, LIGHT = "#2c6fbb", "#d9711c", "#7a7f87", "#d5d8dc"
PX_FS = 30.5 / 16.47 / 0.2998            # fs per time-axis px (dedicated TCAV cal, 16.47 um/um)
PX_MEV = 30.5e-6 / 0.0559 * 10e3         # MeV per energy-axis px (design eta_y 55.9 mm, 10 GeV)
CB3_BUNDLE = REPO.parent / "facet2-twobunch-cnf/src/twobunch_cnf/resources/c_b3_ema/bundle.yaml"
WIDE_BUNDLE = REPO / "models/c_b3w_ema/bundle.yaml"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False})


def render(name):
    return np.load(RES / f"render/{name}.npz")["imgs"][0]


def s2e(probe, idx):
    return np.load(REPO / f"data/{probe}/render_B_{idx:05d}.npz")["plus"]


def fit(name):
    return min(json.loads((RES / f"reach/{name}.json").read_text())["runs"], key=lambda r: r["total_loss"])


def tv(a, b):
    return float(np.abs(a - b).sum())


def show(ax, im, title=None, note=None):
    ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=np.percentile(im, VMAX_PCT))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("0.6")
    if title:
        ax.set_title(title)
    if note:
        ax.text(0.97, 0.05, note, color="w", fontsize=7.5, ha="right", va="bottom", transform=ax.transAxes)


def scale_bars(ax):
    x0, y0 = 20, 16
    ax.plot([x0, x0 + 500 / PX_FS], [y0, y0], color="w", lw=2)
    ax.text(x0 + 250 / PX_FS, y0 + 3, "500 fs", color="w", fontsize=7.5, ha="center", va="bottom")
    ax.plot([x0, x0], [y0, y0 + 200 / PX_MEV], color="w", lw=2)
    ax.text(x0 + 4, y0 + 200 / PX_MEV, "200 MeV", color="w", fontsize=7.5, ha="left", va="top")


def step_label(ax, s):
    ax.set_ylabel(f"L2 scan {0.25 * s:+.1f}°", fontsize=8.5)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(OUT / f"{name}.png")


def fig_pipeline(real):
    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25], hspace=0.25, wspace=0.06)
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.6); ax.axis("off")
    boxes = [(0.1, "Machine settings\n(37 knobs): linac,\nfinal focus, sextupole\nmovers, injector"),
             (2.15, "Beam surrogate\n(conditional flow):\ndrive + witness\nparticles at the TCAV"),
             (4.2, "TCAV + magnets\nto the DTOTR2 screen\n(differentiable model)"),
             (6.25, "Simulated image\nvs real image:\nmismatch (TV)")]
    for x, txt in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.95), 1.75, 1.45, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    fc="#eef3fa", ec=BLUE, lw=1.2))
        ax.text(x + 0.875, 1.675, txt, ha="center", va="center", fontsize=8)
    for x in (1.85, 3.9, 5.95):
        ax.annotate("", (x + 0.3, 1.675), (x, 1.675), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.2))
    ax.add_patch(FancyBboxPatch((8.3, 0.95), 1.6, 1.45, boxstyle="round,pad=0.03,rounding_size=0.12",
                                fc="#fdf1e7", ec=ORANGE, lw=1.2))
    ax.text(9.1, 1.675, "Fit:\nfind the knob errors\nthat minimise\nmismatch + prior", ha="center", va="center",
            fontsize=8)
    ax.annotate("", (8.3, 1.675), (8.0, 1.675), arrowprops=dict(arrowstyle="-|>", color="0.25", lw=1.2))
    ax.annotate("", (0.975, 0.95), (9.1, 0.95), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2,
                                                                 connectionstyle="arc,angleA=-90,angleB=-90,armA=22,armB=22,rad=8"))
    ax.text(5.0, 0.22, "gradients flow back through the screen model and the surrogate", ha="center",
            fontsize=7.5, color=ORANGE)
    for j, s in enumerate(STEPS):
        a = fig.add_subplot(gs[1, j])
        show(a, real[0, s], f"real DTOTR2, L2 scan {0.25 * s:+.1f}°")
        if j == 0:
            scale_bars(a)
            a.set_ylabel("energy  →", fontsize=8)
        a.set_xlabel("time  →", fontsize=8)
    save(fig, "pipeline")


def fig_chain():
    rows = [("No fit: new surrogate at design settings", fit("A1_B")["null"]["img"], LIGHT),
            ("Old surrogate, Stage D's one-sided comparison", fit("X1_D")["final_img_loss"], GRAY),
            ("Old surrogate, comparison fixed", fit("X1_B")["final_img_loss"], GRAY),
            ("New surrogate, linac + final-focus knobs only", fit("A4_B")["final_img_loss"], BLUE),
            ("  + drive/witness charge split", fit("A7_B")["final_img_loss"], BLUE),
            ("  + the other 13 injector and L0B knobs", fit("A1_B")["final_img_loss"], BLUE),
            ("  + TCAV calibration and L2 scan gain", fit("A1calg_B")["final_img_loss"], BLUE),
            ("  + fitted image blur", fit("A1calgr_B")["final_img_loss"], BLUE)]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, c) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=c)
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0.958, color=ORANGE, ls="--", lw=1.2)
    ax.text(0.958, len(rows) - 0.35, "Stage D as published", color=ORANGE, fontsize=7.5, ha="center")
    ax.axvline(0.10, color="k", ls=":", lw=1.2)
    ax.text(0.11, len(rows) - 0.35, "noise floor", fontsize=7.5)
    ax.set_xlim(0, 1.32)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("image mismatch to real (weighted mean TV over 66 images; 0 = identical)")
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GRAY, label="old surrogate (23 knobs, no injector)"),
                       Patch(color=BLUE, label="new surrogate (37 knobs, with injector)")],
              loc="lower right", fontsize=7.5, frameon=False)
    save(fig, "chain")


def fig_gallery(real):
    cols = [("Real DTOTR2", lambda s: real[0, s]),
            ("No fit (design settings)", lambda s: render("null_cb3_B")[s]),
            ("Old surrogate, fitted", lambda s: render("X1_B")[s]),
            ("New surrogate, fitted", lambda s: render("A1_B")[s])]
    fig, axes = plt.subplots(len(STEPS), len(cols), figsize=(9, 5.4))
    for i, s in enumerate(STEPS):
        for j, (title, f) in enumerate(cols):
            im = f(s)
            show(axes[i, j], im, title if i == 0 else None, None if j == 0 else f"TV {tv(im, real[0, s]):.2f}")
        step_label(axes[i, 0], s)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, "gallery")


def fig_widebox():
    cb3 = {k["name"]: k for k in yaml.safe_load(CB3_BUNDLE.read_text())["knobs"]}
    wide = {k["name"]: k for k in yaml.safe_load(WIDE_BUNDLE.read_text())["knobs"]}
    fits = {"original surrogate": (fit("A1calg_B"), cb3, BLUE, "o"),
            "wide-box surrogate": (fit("W1calg_B"), wide, ORANGE, "D")}

    def phys(r, box, name):
        if name in ("L1PhaseSet", "L2PhaseSet"):
            lo, hi = box[name]["low"], box[name]["high"]
            return [lo + u * (hi - lo) for u in r["u_phases"][name]]
        return [box[name]["baseline"] + r["delta_phys"][name]]

    rows = [("Total charge [pC]", "distgen:total_charge:value", (1300, 2200), [(1900, "E331 (~1.9 nC)")]),
            ("Laser pulse separation [ps]", "distgen:t_dist:dists:d2:avg_t:value", (7, 13.6),
             [(13.0, "elog: '13 ps'")]),
            ("L1 phase [deg]", "L1PhaseSet", (-28, -16), []),
            ("L2 phase over the scan [deg]", "L2PhaseSet", (-42, -27), [])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 4.6))
    for ax, (lab, name, lim, refs) in zip(axes, rows):
        ax.barh(0, wide[name]["high"] - wide[name]["low"], left=wide[name]["low"], height=0.9, color=LIGHT,
                label="wide training box")
        ax.barh(0, cb3[name]["high"] - cb3[name]["low"], left=cb3[name]["low"], height=0.9, color="#9aa0a8",
                label="old training box")
        for k, (lab_f, (r, box, c, m)) in enumerate(fits.items()):
            v = phys(r, box, name)
            yy = 0.18 if k == 0 else -0.18
            if len(v) == 2:
                ax.plot(v, [yy, yy], color=c, lw=2.2, solid_capstyle="butt", label=f"fit with the {lab_f}")
                ax.plot(v, [yy, yy], color=c, ls="none", marker="|", ms=9, mew=2)
            else:
                ax.plot(v, [yy], color=c, ls="none", marker=m, ms=6, label=f"fit with the {lab_f}")
        for x, t in refs:
            ax.axvline(x, color="k", lw=1, ls=":")
            ax.text(x, 0.62, f"  {t}", fontsize=7, ha="left", va="bottom")
        ax.set_xlim(*lim); ax.set_ylim(-0.6, 0.95)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title(lab, loc="left", fontsize=8, pad=2)
    h, lab = axes[0].get_legend_handles_labels()
    order = [2, 3, 0, 1]
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc="upper center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(hspace=0.9)
    save(fig, "widebox")


def fig_rowcorr(real):
    unc, cor = render("null_cb3_D"), render("null_cb3_B")
    cols = [("Real DTOTR2 (row-corrected)", lambda s: real[0, s], None),
            ("Model, uncorrected\n(as Stage D compared it)", lambda s: unc[s], None),
            ("Model, same row correction\n(Stage D2)", lambda s: cor[s], lambda s: f"change TV {tv(cor[s], unc[s]):.2f}")]
    fig, axes = plt.subplots(len(STEPS), len(cols), figsize=(7, 5.6))
    for i, s in enumerate(STEPS):
        for j, (title, f, note) in enumerate(cols):
            show(axes[i, j], f(s), title if i == 0 else None, note(s) if note else None)
        step_label(axes[i, 0], s)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, "rowcorr")


def fig_gain(real):
    met = json.loads((RES / "metrology_A1calgr_B_metro.json").read_text())
    full = [r["l2_gain"] for r in met["restarts"]]
    est = [("Whole scan (3 restarts)", np.mean(full), BLUE),
           ("Even steps only", met["splits"]["even"]["l2_gain"], BLUE),
           ("Odd steps only", met["splits"]["odd"]["l2_gain"], BLUE),
           ("Low-compression half (0 to +4°)", met["splits"]["low"]["l2_gain"], BLUE),
           ("Deep-compression half (+4.25 to +8°)", met["splits"]["deep"]["l2_gain"], BLUE),
           ("Check: fit to simulated images made with g = 1", fit("A1g_B_theft_lam1e-2")["l2_gain"], GRAY)]
    fig = plt.figure(figsize=(9, 5.4))
    top, bot = fig.subfigures(2, 1, height_ratios=[1.25, 0.75], hspace=0.04)
    ax = top.subplots()
    y = np.arange(len(est))[::-1]
    ax.axvspan(0.93, 1.09, color=ORANGE, alpha=0.18, lw=0)
    ax.text(1.01, len(est) - 0.45, "predicted from the\nmeasured klystron setup", color=ORANGE, fontsize=7.5,
            ha="center", va="bottom")
    ax.axvline(1.0, color="0.4", lw=0.8)
    for yi, (lab, v, c) in zip(y, est):
        ax.plot(v, yi, "o", color=c, ms=7)
        ax.text(v - 0.006, yi + 0.18, f"{v:.2f}", fontsize=7.5, ha="right", va="bottom")
    ax.set_yticks(y, [e[0] for e in est]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.7, 1.15); ax.set_ylim(-0.6, len(est) + 0.4)
    ax.set_xlabel("fitted L2 scan gain g (real L2 phase step = g × set step)")
    ax.grid(axis="x", alpha=0.3)
    cols = [("Real DTOTR2", real[0, DEEP]),
            ("Full simulation (S2E),\ndesign settings", s2e("s2e_probe_15673_v1", 6)),
            ("Full simulation (S2E),\nfitted settings", s2e("s2e_probe_15673_v3_heater_delay", 51)),
            ("Surrogate fit", render("A1_B")[DEEP])]
    axs = bot.subplots(1, len(cols))
    for a, (title, im) in zip(axs, cols):
        show(a, im, title, None if a is axs[0] else f"TV {tv(im, real[0, DEEP]):.2f}")
    axs[0].set_ylabel(f"L2 scan {0.25 * DEEP:+.1f}°", fontsize=8.5)
    bot.subplots_adjust(wspace=0.05, left=0.125, right=0.9, top=0.78, bottom=0.02)
    save(fig, "gain")


def fig_thickness(real):
    from stageD2_blur_figure import s2e_rows
    from stageD2_np1e6_figure import global_blur
    rows = s2e_rows()
    gb = global_blur(rows, real, (2, 10, 16, 26, 29, 32))
    se, st = gb["sigma_E_px"], gb["sigma_t_px"]

    def blurred(s):
        b = gaussian_filter(rows[s]["plus"], (se, st), mode="constant")
        return b / b.sum()

    cols = [("Real DTOTR2", lambda s: real[0, s]),
            ("Full simulation (S2E),\nfitted settings", lambda s: rows[s]["plus"]),
            (f"S2E blurred: {st * 30.5:.0f} µm along time,\n{se * 30.5:.0f} µm along energy", blurred)]
    fig, axes = plt.subplots(len(STEPS), len(cols), figsize=(7, 5.6))
    for i, s in enumerate(STEPS):
        for j, (title, f) in enumerate(cols):
            im = f(s)
            show(axes[i, j], im, title if i == 0 else None, None if j == 0 else f"TV {tv(im, real[0, s]):.2f}")
        step_label(axes[i, 0], s)
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    save(fig, "thickness")


PLAIN = {"l2_gain": "L2 scan gain", "l1_shift": "L1 phase offset", "l2_shift": "L2 phase offset",
         "drive_fraction": "drive charge fraction", "L1EnergyOffset": "L1 energy", "L2EnergyOffset": "L2 energy",
         "L3EnergyOffset": "L3 energy", "GUNF:rf_field_scale": "gun field", "GUNF:theta0_deg": "gun phase",
         "laser_timing_ps": "laser timing", "L0BPhaseSet": "L0B phase"}


def plain(name):
    if name in PLAIN:
        return PLAIN[name]
    if name.endswith("Offset") and name[:2] in ("S1", "S2"):
        return f"mover {name[:4]} {name[5]}"
    return name.replace("kG", "")


def fig_metrology():
    met = json.loads((RES / "metrology_A1calgr_B_metro.json").read_text())
    reg = json.loads((RES / "register_A1calgr_B_metro_r0.json").read_text())
    names = reg["param_names"]
    labels = []
    for k, v in enumerate(reg["kept_directions"]):
        top = np.argsort(np.abs(v))[::-1][:2]
        sg = ["+" if v[t] > 0 else "−" for t in top]
        labels.append(f"{k + 1}: {sg[0]}{plain(names[top[0]])}  {sg[1]}{plain(names[top[1]])}")
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, sharex=True)
    for ax, (a, b, title) in zip(axes, [("even", "odd", "Control: fit on even vs odd steps"),
                                        ("low", "deep", "Test: fit on low vs deep compression")]):
        pa, pb = np.array(met["splits"][a]["proj"]), np.array(met["splits"][b]["proj"])
        for yi, u, v in zip(y, pa, pb):
            ax.plot([u, v], [yi, yi], color="0.7", lw=1.2, zorder=1)
        ax.plot(pa, y, "o", color=BLUE, ms=6, label=f"{a} steps" if a == "even" else "low-compression half", zorder=2)
        ax.plot(pb, y, "D", color=ORANGE, ms=5, label=f"{b} steps" if b == "odd" else "deep-compression half", zorder=2)
        rms = met["splits"][f"{a}_vs_{b}"]["proj_diff_rms"]
        ax.set_title(f"{title}\n(rms difference {rms:.2f} prior σ)")
        ax.axvline(0, color="0.4", lw=0.8)
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlabel("fitted value of the combination [prior σ]")
        ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    axes[0].set_yticks(y, labels); axes[0].tick_params(axis="y", length=0)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "metrology")


def fig_ruledout(real):
    base = s2e("s2e_probe_15673_v3_heater_delay", 51)
    cols = [("Real DTOTR2", real[0, DEEP]),
            ("S2E, fitted settings", base),
            ("+ laser heater, 800 µJ", s2e("s2e_probe_15673_v3_heater_delay", 62)),
            ("3D CSR in the bends", s2e("s2e_csr3d_allb_hi", 1)),
            ("CSR off", s2e("s2e_csr3d_nocsr_hi", 1))]
    fig, axes = plt.subplots(1, len(cols), figsize=(11, 1.9))
    for j, (title, im) in enumerate(cols):
        show(axes[j], im, title, None if j == 0 else f"TV {tv(im, real[0, DEEP]):.2f}")
    axes[0].set_ylabel(f"L2 scan {0.25 * DEEP:+.1f}°", fontsize=8.5)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "ruledout")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    real = targets("B")["images"].numpy()
    fig_pipeline(real)
    fig_chain()
    fig_gallery(real)
    fig_widebox()
    fig_rowcorr(real)
    fig_gain(real)
    fig_thickness(real)
    fig_metrology()
    fig_ruledout(real)


if __name__ == "__main__":
    main()
