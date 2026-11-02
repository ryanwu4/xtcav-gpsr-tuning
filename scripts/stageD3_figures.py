"""Stage D3 figures and summary: does the model error depend on the machine setpoint?

Reads results/stageD3/{step_selection.json, fits/*.json, render/*.npz} (CPU only):

  error_by_setpoint.png   per-step image error (TV, mean of the two streak signs) against the real
                          bunch separation, one colour per run: the no-fit null, each run fitted
                          alone on its whole scan, and the joint window fit scored on whole scans
  transfer.png            each run's window error: fitted alone, fitted jointly, and predicted
                          from the other two runs (only its own L1/L2 refit)
  gallery.png             real vs joint fit, + streak, three window steps + one deep step per run
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
    p = OUT / "fits" / f"{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


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
    panels = [("no fit (golden machine at the readback phases)", lambda r: load("N_full")),
              ("each run fitted alone, whole scan", single_full),
              ("joint fit on the windows, scored on whole scans", lambda r: load("J_win"))]
    panels = [(t, f) for t, f in panels if any(f(r) is not None for r in ORDER)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.3), sharey=True, squeeze=False)
    for ax, (title, getter) in zip(axes[0], panels):
        for r in ORDER:
            fit = getter(r)
            if fit is None:
                continue
            steps, tv, fl = per_step(fit, r)
            s = sel[DS[r]]
            sep = np.asarray(s["real_plus"])[steps]
            keep = np.asarray(s["keep"])[steps]
            pre = steps <= s["i_min"]
            c = RUN_COLORS[DS[r]]
            o = np.argsort(-sep[pre])
            ax.plot(sep[pre][o], tv[pre][o], "-", color=c, lw=1.0, alpha=0.6)
            ax.plot(sep[pre & keep], tv[pre & keep], "o", color=c, ms=5.5, label=LABEL[r])
            ax.plot(sep[pre & ~keep], tv[pre & ~keep], "o", mfc="white", mec=c, ms=5.5)
        ax.axvline(FLOOR_UM, color="#6B7689", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("real bunch separation [µm]  (← less compressed | more compressed →)")
    axes[0, 0].set_ylabel("image error (TV, 0 = identical, 2 = disjoint)")
    axes[0, 0].legend(fontsize=8, loc="upper left", title="filled = in the fit window", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "error_by_setpoint.png", dpi=140)
    plt.close(fig)


def window_tv(fit, run, sel, key="score_full"):
    steps, tv, _ = per_step(fit, run)
    keep = np.asarray(sel[DS[run]]["keep"])[steps]
    return float(tv[keep].mean())


def figure_transfer(sel, summary):
    rows = []
    for r in ORDER:
        vals = {}
        for name, arm in (("no fit", "N_full"), ("fitted alone", f"S{r}_win"),
                          ("fitted jointly", "J_win"), ("predicted from the other two", f"LORO{r}")):
            f = load(arm)
            if f is not None:
                vals[name] = window_tv(f, r, sel)
        rows.append((r, vals))
    summary["window_tv"] = {r: v for r, v in rows}
    names = ["no fit", "fitted alone", "fitted jointly", "predicted from the other two"]
    colors = ["#B8BEC9", "#6B7689", "#2a78d6", "#eb6834"]
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    w = 0.2
    for k, (n, c) in enumerate(zip(names, colors)):
        x = [i + (k - 1.5) * w for i, (r, v) in enumerate(rows) if n in v]
        y = [v[n] for r, v in rows if n in v]
        ax.bar(x, y, w * 0.92, color=c, label=n)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([LABEL[r] for r, _ in rows], fontsize=8.5)
    ax.set_ylabel("mean image error in the window (TV)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "transfer.png", dpi=140)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sel = json.loads((OUT / "step_selection.json").read_text())
    summary = {}
    figure_error(sel)
    figure_transfer(sel, summary)
    for arm in ("J_win", "J_full", "J_win_drift") + tuple(f"S{r}_{m}" for r in ORDER for m in ("win", "full")):
        f = load(arm)
        if f is None:
            continue
        b = min(f["fits"], key=lambda x: x["final_img_loss"])
        summary[arm] = {"img": b["final_img_loss"], "l2_gain": b["l2_gain"], "per_run": b["per_run"],
                        "walls": b["walls"], "blur_px": b["blur_px"],
                        "restarts": [x["final_img_loss"] for x in f["fits"]],
                        "delta_phys": b["delta_phys"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (v if k == "window_tv" else {kk: v[kk] for kk in ("img", "l2_gain", "walls")})
                      for k, v in summary.items()}, indent=1, default=float))


if __name__ == "__main__":
    main()
