"""Stage D3 figures and summary: does the model error depend on the machine setpoint?

Reads results/stageD3/{step_selection.json, fits/*.json, render/*.npz} (CPU only):

  error_by_setpoint.png   per-step image error (TV, mean of the two streak signs) over the whole
                          scans, x aligned at each run's real full compression, one colour per run:
                          each run fitted alone, one machine on all whole scans, one machine on
                          the windows only
  transfer.png            each run's window error: fitted alone, fitted jointly, and predicted
                          from the other two runs (only its own L1/L2 refit)
  gallery.png             real vs joint fit, + streak: each run's last window step and its full-
                          compression step (held out)
  blen.png                bunch-length monitors along each scan: real BLEN after BC14 and in the FF
                          (raw counts) vs the FACET2-S2E coherent-power proxy at the neighbouring
                          markers (stageD3_blen_proxy.py), each normalized to its scan maximum
  -> results/stageD3/summary.json

  PYTHONPATH=$PWD/src ~/miniconda3/envs/slac-rl/bin/python scripts/stageD3_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stageD3_step_select import FLOOR_UM, RUN_COLORS  # noqa: E402

OUT = REPO / "results" / "stageD3"
FIG = OUT / "figures"
DS = {"15668": "TEST_15668", "15671": "E300_15671", "15673": "E300_15673"}
LABEL = {"15671": "15671: deep compression (L1 ≈ −26°)", "15668": "15668: middle (L1 ≈ −23.7°)",
         "15673": "15673: least compressed (L1 ≈ −20°)"}
ORDER = ("15671", "15668", "15673")   # deep -> least compressed


def load(arm):
    """The arm's fit file with the lowest best-restart image loss (extra restarts live in <arm>_s<k>.json),
    with every restart's loss gathered under 'all_restarts'."""
    files = [OUT / "fits" / f"{arm}.json"] + sorted((OUT / "fits").glob(f"{arm}_s[0-9]*.json"))
    fits = [json.loads(f.read_text()) for f in files if f.exists()]
    fits = [f for f in fits if "score_full" in f]
    if not fits:
        return None
    key = lambda f: (f["held_out_fit"]["final_img_loss"] if "held_out_fit" in f
                     else min(x["final_img_loss"] for x in f.get("fits", [{"final_img_loss": 0.0}])))
    best = min(fits, key=key)
    best["all_restarts"] = [x["final_img_loss"] for f in fits for x in f.get("fits", [])]
    return best


def per_step(fit, run):
    """(steps, TV mean over signs, floor mean over signs) of one run in a fit's full-scan score."""
    sc = fit["score_full"]
    tv = np.asarray(sc["per_image_tv"]).mean(0)
    fl = np.nanmean(np.asarray(sc["floors"]), 0)
    rows = [i for i, q in enumerate(sc["meta"]) if q["run"] == run]
    return np.array([sc["meta"][i]["step"] for i in rows]), tv[rows], fl[rows]


def single_full(run):
    return load(f"S{run}_full")


def figure_error(sel):
    """Per-step image error over the whole scans, on an axis aligned at each run's full compression on
    the screen (x = L2 phase past the real separation minimum). Triangles: where the real BC14 monitor
    peaks (bunch shortest after BC14)."""
    real = real_blen()
    panels = [("each run fitted alone on its whole scan", single_full),
              ("one machine fitted to all three whole scans", lambda r: load("J_full")),
              ("one machine fitted to the windows only", lambda r: load("J_win"))]
    panels = [(t, f) for t, f in panels if any(f(r) is not None for r in ORDER)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 4.7), sharey=True, squeeze=False)
    for ax, (title, getter) in zip(axes[0], panels):
        for r in ORDER:
            fit = getter(r)
            if fit is None:
                continue
            steps, tv, fl = per_step(fit, r)
            s = sel[DS[r]]
            x = 0.25 * (steps - s["i_min"])
            keep = np.asarray(s["keep"])[steps]
            c = RUN_COLORS[DS[r]]
            ax.plot(x, tv, "-", color=c, lw=1.0, alpha=0.6)
            ax.plot(x[keep], tv[keep], "o", color=c, ms=5.0, label=LABEL[r])
            ax.plot(x[~keep], tv[~keep], "o", mfc="white", mec=c, ms=5.0)
            pk = 0.25 * (int(np.nanargmax(real[DS[r]]["BC14"])) - s["i_min"])
            ax.plot(pk, 1.52, "v", color=c, ms=8)
        ax.axvline(0, color="#6B7689", lw=0.8, ls="--")
        ax.text(0.08, 0.36, "screen: full\ncompression", color="#6B7689", fontsize=7.5)
        ax.text(-6.9, 1.555, "▼ bunch shortest after BC14 (real monitor)", color="#6B7689", fontsize=7.5)
        ax.set_ylim(0.3, 1.62)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("L2 phase past full compression on the screen [deg]")
    axes[0, 0].set_ylabel("image error (TV, 0 = identical, 2 = disjoint)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               title="filled = window steps; open = outside the window (fitted in the left and middle "
                     "panels, held out in the right panel)", title_fontsize=8.5)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "error_by_setpoint.png", dpi=140)
    plt.close(fig)


def window_tv(fit, run, sel, key="score_full"):
    steps, tv, _ = per_step(fit, run)
    keep = np.asarray(sel[DS[run]]["keep"])[steps]
    return float(tv[keep].mean())


def figure_transfer(sel, summary):
    arms = (("no fit", "N_full"), ("fitted alone", "S{r}_win"), ("fitted jointly", "J_win"),
            ("fitted jointly, own L2 gain per run", "J_win_g"), ("predicted from the other two", "LORO{r}"),
            ("predicted from the other two, own L2 gain", "LOROg{r}"))
    style = {"no fit": ("#B8BEC9", None), "fitted alone": ("#6B7689", None),
             "fitted jointly": ("#2a78d6", None), "fitted jointly, own L2 gain per run": ("#2a78d6", "///"),
             "predicted from the other two": ("#eb6834", None),
             "predicted from the other two, own L2 gain": ("#eb6834", "///")}
    rows = []
    for r in ORDER:
        vals = {}
        for name, arm in arms:
            f = load(arm.format(r=r))
            if f is not None:
                vals[name] = window_tv(f, r, sel)
        rows.append((r, vals))
    summary["window_tv"] = {r: v for r, v in rows}
    names = [n for n, _ in arms if any(n in v for _, v in rows)]
    fig, ax = plt.subplots(figsize=(10.0, 4.3))
    w = 0.84 / len(names)
    for k, n in enumerate(names):
        c, hatch = style[n]
        x = [i + (k - (len(names) - 1) / 2) * w for i, (r, v) in enumerate(rows) if n in v]
        y = [v[n] for r, v in rows if n in v]
        ax.bar(x, y, w * 0.9, color="white" if hatch else c, edgecolor=c, hatch=hatch, lw=1.2, label=n)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([LABEL[r] for r, _ in rows], fontsize=8.5)
    ax.set_ylabel("mean image error in the window (TV)")
    ax.set_ylim(0, 1.3 * max(max(v.values()) for _, v in rows if v))
    ax.legend(fontsize=7.5, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIG / "transfer.png", dpi=140)
    plt.close(fig)


TARGETS = {"15668": "results/stageD3/15668/targets_rowref_clean.npz",
           "15671": "results/stageD3/15671/targets_rowref_clean.npz",
           "15673": "results/stageD2/15673/targets_rowref_clean.npz"}


def real_plus(run):
    """Real + streak targets in the fit's frame (rows flipped, signs swapped: flip1_swap1)."""
    im = np.load(REPO / TARGETS[run])["images"]          # (step, sign[plus, neg], ny, nx)
    return np.flip(im, -2)[:, ::-1][:, 0]                 # model sign +1 = camera 'neg', flipped


def figure_gallery(sel, arm="J_win"):
    p = OUT / "render" / f"{arm}.npz"
    if not p.exists():
        return
    d = np.load(p)
    imgs, meta = d["imgs"], json.loads(str(d["meta"]))
    cols = []
    for r in ORDER:
        s = sel[DS[r]]
        cols.append((r, s["kept_steps"][-1], "last fitted step"))
        cols.append((r, s["i_min"], "full compression (not fitted)"))
    fig, axes = plt.subplots(2, len(cols), figsize=(2.15 * len(cols), 3.9))
    for j, (r, st, what) in enumerate(cols):
        i = next(k for k, q in enumerate(meta) if q["run"] == r and q["step"] == st)
        real = real_plus(r)[st]
        model = imgs[0, i]
        tv = float(np.abs(model - real).sum())
        for row, (im, lab) in enumerate(((real, "real"), (model, "joint fit"))):
            ax = axes[row, j]
            ax.imshow(im, origin="lower", aspect="auto", cmap="viridis", vmin=0,
                      vmax=np.percentile(im, 99.7))
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(lab, fontsize=9)
        axes[1, j].text(0.97, 0.04, f"TV {tv:.2f}", color="w", fontsize=7.5, ha="right", va="bottom",
                        transform=axes[1, j].transAxes)
        axes[0, j].set_title(f"{r}, scan {0.25 * st:+.2f}°\n{what}", fontsize=7.5,
                             color=RUN_COLORS[DS[r]])
    fig.tight_layout()
    fig.savefig(FIG / "gallery.png", dpi=140)
    plt.close(fig)


XTCAV = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
BLEN_PV = {"BC11": ("BSA_List_S11", "BLEN_LI11_359_BRAW"), "BC14": ("BSA_List_S14", "BLEN_LI14_888_BRAW"),
           "FF": ("BSA_List_S20", "BLEN_LI20_3014_BRAW")}
BLEN_MARKER = {"BC11": "ENDBC11_2", "BC14": "ENDBC14_2", "FF": "MFFF"}
BLEN_SIM = {"E300_15671": "E300_15671_l2ref-38.00", "TEST_15668": "TEST_15668_l2ref-40.35",
            "E300_15673": "E300_15673_l2ref-38.00"}     # the July sims the step selection used


def real_blen():
    """Per-step medians of the three BLEN monitors (cached: the DAQ .mat files are slow to read)."""
    cache = OUT / "blen_real.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import scipy.io as sio
    out = {}
    for ds in DS.values():
        d = sio.loadmat(str(XTCAV / ds / f"{ds}.mat"), simplify_cells=True)["data_struct"]
        sc = d["scalars"]
        st = np.asarray(sc["steps"], int)
        out[ds] = {k: [float(np.nanmedian(np.asarray(sc[g][pv], float)[st == s])) for s in np.unique(st)]
                   for k, (g, pv) in BLEN_PV.items()}
        r = sc["BSA_List_S11RF"]
        l1 = np.nanmean([np.asarray(r["KLYS_LI11_11_FB_FAST_PACT"], float),
                         np.asarray(r["KLYS_LI11_21_FB_FAST_PACT"], float)], 0)
        out[ds]["L1_readback"] = [float(np.nanmedian(l1[st == s])) for s in np.unique(st)]
    cache.write_text(json.dumps(out, indent=1))
    return out


def figure_blen(sel, summary, proxy="int_lambda2_per_mm"):
    real = real_blen()
    mons = ("BC14",)   # the FF pyrometer barely moves along any real scan, so its shape says nothing
    fig, axes = plt.subplots(len(mons), 3, figsize=(13.0, 3.9 * len(mons)), sharex=True, sharey="row",
                             squeeze=False)
    out = {}
    for j, r in enumerate(ORDER):
        ds = DS[r]
        s = sel[ds]
        p = OUT / "blen_proxy" / f"{BLEN_SIM[ds]}.json"
        sim = json.loads(p.read_text()) if p.exists() else None
        scan = 0.25 * np.arange(len(s["real_plus"]))
        kept = s["kept_steps"]
        out[r] = {}
        for i, mon in enumerate(mons):
            ax = axes[i, j]
            ax.axvspan(scan[kept[0]] - 0.12, scan[kept[-1]] + 0.12, color="#1baf7a", alpha=0.12, lw=0)
            y = np.asarray(real[ds][mon])
            ax.plot(scan, y / np.nanmax(y), "o-", color="#2a78d6", ms=3.5, lw=1.2, label="real monitor")
            ax.axvline(scan[s["i_min"]], color="#2a78d6", ls="--", lw=0.9)
            out[r][f"real_{mon}_peak_deg"] = float(scan[int(np.nanargmax(y))])
            if sim is not None:
                st = sorted(int(k) for k in sim["steps"])
                xs = np.array([sim["steps"][str(k)]["scan_deg"] for k in st]) - s["delta_deg"]["plus"]
                v = np.array([sim["steps"][str(k)][BLEN_MARKER[mon]][proxy] for k in st])
                ax.plot(xs, v / v.max(), "s-", color="#eb6834", ms=3.0, lw=1.2, label="simulation (aligned)")
                out[r][f"sim_{mon}_peak_deg"] = float(xs[int(np.argmax(v))])
            if i == 0:
                ax.set_title(LABEL[r], fontsize=10, color=RUN_COLORS[ds])
            if i == len(mons) - 1:
                ax.set_xlabel("L2 phase scan value [deg]")
        axes[0, j].text(scan[s["i_min"]] + 0.1, 0.03, "screen: full\ncompression", color="#2a78d6", fontsize=7.5,
                        va="bottom")
    axes[0, 0].set_ylabel("bunch-length signal after BC14\n(signal / scan max)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axes.flat:
        ax.set_ylim(0, 1.08)
    fig.text(0.5, 0.01, "real: raw BLEN counts (higher = shorter bunch).  simulation: coherent-power proxy "
             "∫λ(z)² dz of the FACET2-S2E beam at the monitor, shifted in L2 as in the step selection.  "
             "green = fit window", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "blen.png", dpi=140)
    plt.close(fig)
    summary["blen_peaks"] = out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sel = json.loads((OUT / "step_selection.json").read_text())
    summary = {}
    figure_error(sel)
    figure_transfer(sel, summary)
    figure_gallery(sel)
    figure_blen(sel, summary)
    arms = (("J_win", "J_full", "J_win_drift", "J_win_g") + tuple(f"LORO{r}" for r in ORDER)
            + tuple(f"LOROg{r}" for r in ORDER)
            + tuple(f"S{r}_{m}" for r in ORDER for m in ("win", "full")))
    for arm in arms:
        f = load(arm)
        if f is None:
            continue
        b = min(f["fits"], key=lambda x: x["final_img_loss"])
        summary[arm] = {"img": b["final_img_loss"], "l2_gain": b["l2_gain"], "per_run": b["per_run"],
                        "walls": b["walls"], "blur_px": b["blur_px"], "restarts": f["all_restarts"],
                        "delta_phys": b["delta_phys"]}
        if "held_out_fit" in f:
            h = f["held_out_fit"]
            summary[arm]["held_out"] = {"run": f["held_out"], "img_before_refit": h["img_before_refit"],
                                        "img_after_refit": h["final_img_loss"], "per_run": h["per_run"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (v if "img" not in v else {kk: v[kk] for kk in ("img", "l2_gain", "walls")})
                      for k, v in summary.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
