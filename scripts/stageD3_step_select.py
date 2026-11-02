"""Stage D3 step selection: which L2 scan steps of the three 2026-04-15 E300 scans to fit.

Uses the 2026-07-23 sim2real study as it stands: real Tier-0 energy-gated bunch separation
(per-shot medians) against the matched FACET2-S2E Tier-1 scans, aligned with the study's one
fitted L2 offset per run (refit here on the pre-minimum branch). A step PASSES when, on the
+ streak (the study's primary pairing):

  1. bands separable   real median separation >= FLOOR_UM (both the gate and the bi-Gaussian
                       degrade below ~50 um, addendum 1) and the step is before the real
                       separation minimum (full compression; past it the curve folds back)
  2. matched           |sim - real| <= max(ABS_TOL_UM, REL_TOL * real)  (REL_TOL ~ the 15%
                       real streak-calibration systematic)

The fitted WINDOW (Ryan, 2026-11-01: one window per run) runs from the scan start to the last
passing step before the first sustained divergence (>= RUN_FAIL consecutive failing steps);
isolated 1-2 step failures inside it are median noise and stay in. The local sim/real slope
ratio and the - streak agreement are reported, not used to select (the real - streak gate
fails mid-scan in 15673, where the bands overlap in energy).

Sim runs: 15671 = the sigma_z-anchored rerun (L1 -22.35), 15668 = the aligned rerun at
L2_ref -40.35 (the -38 run never reached the scan's first 10 steps), 15673 = the July run.

  PYTHONPATH=$PWD/src ~/miniconda3/envs/slac-rl/bin/python scripts/stageD3_step_select.py
-> results/stageD3/step_selection.json, results/stageD3/figures/{step_selection,s2e_mismatch}.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
S2R = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results")
OUT = REPO / "results" / "stageD3"
RUNS = {  # dataset: (sim run tag, L2 ref of that sim run [deg])
    "E300_15671": ("l1sigma370_full", -38.0),
    "TEST_15668": ("l2ref-40.35_value_match", -40.35),
    "E300_15673": ("l2ref-38.00_value_match", -38.0),
}
TITLES = {"E300_15671": "E300_15671: deep compression (L1 ≈ −26°)",
          "TEST_15668": "TEST_15668: middle (L1 ≈ −23.7°)",
          "E300_15673": "E300_15673: least compressed (L1 ≈ −20°)"}
FLOOR_UM, ABS_TOL_UM, REL_TOL, SLOPE_FACTOR, RUN_FAIL = 50.0, 20.0, 0.15, 2.0, 3


def per_step(v, step, sign, n_steps, s):
    med, lo, hi = (np.full(n_steps, np.nan) for _ in range(3))
    for i in range(n_steps):
        x = v[(step == i + 1) & (sign == s)]
        x = x[np.isfinite(x)]
        if len(x):
            lo[i], med[i], hi[i] = np.percentile(x, [25, 50, 75])
    return med, lo, hi


def sim_curve(S, sign):
    sel = S["streak_sign"] == sign
    o = np.argsort(S["scan_val"][sel])
    return S["scan_val"][sel][o], S["eg_sep_median_um"][sel][o]


def align(sv, real, ssv, ssep, mask, bracket=4.0, grid=0.05):
    """The July fit_shift: delta minimizing mean (real(s) - sim(s + delta))^2 on mask."""
    ok = np.isfinite(ssep)
    ssv, ssep = ssv[ok], ssep[ok]
    best, cost = 0.0, np.inf
    for d in np.arange(-bracket, bracket + 1e-9, grid):
        si = np.interp(sv, ssv - d, ssep, left=np.nan, right=np.nan)
        m = mask & np.isfinite(real) & np.isfinite(si)
        if m.sum() >= 6 and np.mean((real[m] - si[m]) ** 2) < cost:
            best, cost = float(d), float(np.mean((real[m] - si[m]) ** 2))
    return best


def local_slope(sv, y, half=2):
    """d y / d sv from a line fit over the 2*half+1 neighbouring finite points."""
    out = np.full(len(sv), np.nan)
    for i in range(len(sv)):
        j = np.arange(max(0, i - half), min(len(sv), i + half + 1))
        j = j[np.isfinite(y[j])]
        if len(j) >= 3:
            out[i] = np.polyfit(sv[j], y[j], 1)[0]
    return out


def select(ds, tag, l2ref_sim, floor=FLOOR_UM, abs_tol=ABS_TOL_UM, rel_tol=REL_TOL,
           slope_factor=SLOPE_FACTOR):
    R = np.load(S2R / "tier0_real_features" / ds / "features.npz")
    rs = json.loads((S2R / "tier0_real_features" / ds / "summary.json").read_text())
    sv = np.asarray(rs["scan"]["vals"], float)
    n = len(sv)
    S = np.load(S2R / "tier1_sim_scans" / ds / tag / "features.npz")
    real, sim = {}, {}
    for s, nm in ((1.0, "plus"), (-1.0, "neg")):
        real[nm] = per_step(R["eg_sep_median_um"], R["step"], R["streak_sign"], n, s)
    i_min = int(np.nanargmin(real["plus"][0]))
    pre = np.arange(n) <= i_min
    deltas = {}
    for s, nm in ((1.0, "plus"), (-1.0, "neg")):
        ssv, ssep = sim_curve(S, s)
        d = align(sv, real["plus"][0], ssv, ssep, pre)
        deltas[nm] = d
        sim[nm] = np.interp(sv, ssv - d, ssep, left=np.nan, right=np.nan)
    rp, sp = real["plus"][0], sim["plus"]
    c_floor = pre & (rp >= floor)
    tol = np.maximum(abs_tol, rel_tol * rp)
    c_match = np.isfinite(sp) & (np.abs(sp - rp) <= tol)
    k_real, k_sim = local_slope(sv, rp), local_slope(sv, sp)
    ratio = k_sim / k_real
    c_slope = np.isfinite(ratio) & (ratio >= 1.0 / slope_factor) & (ratio <= slope_factor)
    passed = c_floor & c_match
    fails = np.flatnonzero(~passed)
    end = n - 1
    for i in fails:  # first run of RUN_FAIL consecutive failures ends the window
        if np.all(~passed[i:i + RUN_FAIL]) and i + RUN_FAIL <= n:
            end = i - 1
            break
    keep = np.zeros(n, bool)
    keep[:end + 1] = True
    keep &= np.arange(n) <= max([j for j in range(end + 1) if passed[j]], default=-1)
    # the - streak, for the record (same tolerance, its own alignment)
    rn, sn = real["neg"][0], sim["neg"]
    neg_match = np.isfinite(sn) & np.isfinite(rn) & (np.abs(sn - rn) <= np.maximum(abs_tol, rel_tol * rn))
    reason = []
    for i in range(n):
        r = [] if c_floor[i] else (["past full compression"] if not pre[i] else ["below 50 µm floor"])
        if not np.isfinite(sp[i]):
            r.append("no sim")
        elif not c_match[i]:
            r.append(f"sim off by {sp[i] - rp[i]:+.0f} µm")
        if np.isfinite(sp[i]) and not c_slope[i]:
            r.append(f"(slope ratio {ratio[i]:.2f})" if np.isfinite(ratio[i]) else "(slope n/a)")
        reason.append(("in window" if keep[i] else "out") + (": " + "; ".join(r) if r else ""))
    return {"dataset": ds, "sim_tag": tag, "scan_vals_deg": sv.tolist(),
            "l2ref_real_deg": l2ref_sim + deltas["plus"], "delta_deg": deltas,
            "i_min": i_min, "real_plus": rp.tolist(),
            "real_plus_iqr": [real["plus"][1].tolist(), real["plus"][2].tolist()],
            "real_neg": rn.tolist(), "sim_plus": sp.tolist(), "sim_neg": sn.tolist(),
            "slope_ratio": ratio.tolist(), "passed": passed.tolist(), "keep": keep.tolist(),
            "neg_match": neg_match.tolist(), "reason": reason,
            "kept_steps": [int(i) for i in np.flatnonzero(keep)],
            "kept_scan_deg": [float(sv[i]) for i in np.flatnonzero(keep)],
            "window_slope_ratio": _window_slope_ratio(sv, rp, sp, keep),
            "criteria": {"floor_um": floor, "abs_tol_um": abs_tol, "rel_tol": rel_tol,
                         "run_fail": RUN_FAIL, "slope_factor_reported": slope_factor}}


def _window_slope_ratio(sv, rp, sp, keep):
    """sim/real d(sep)/d(L2) over the window (the July study's Jacobian check)."""
    m = keep & np.isfinite(rp) & np.isfinite(sp)
    if m.sum() < 3:
        return None
    return float(np.polyfit(sv[m], sp[m], 1)[0] / np.polyfit(sv[m], rp[m], 1)[0])


def figure(res, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (ds, r) in zip(axes, res.items()):
        sv = np.asarray(r["scan_vals_deg"])
        keep = np.asarray(r["keep"])
        rp, sp = np.asarray(r["real_plus"]), np.asarray(r["sim_plus"])
        lo, hi = (np.asarray(a) for a in r["real_plus_iqr"])
        for i in np.flatnonzero(keep):
            ax.axvspan(sv[i] - 0.125, sv[i] + 0.125, color="#2E9E5B", alpha=0.13, lw=0)
        ax.fill_between(sv, lo, hi, color="#2563EB", alpha=0.18, lw=0)
        ax.plot(sv, rp, "o-", color="#2563EB", ms=4, label="real (median, IQR)")
        ax.plot(sv, sp, "s-", color="#EA8600", ms=4, label="simulation (aligned)")
        ax.axhline(FLOOR_UM, color="#6B7689", lw=0.8, ls="--")
        ax.text(sv[-1], FLOOR_UM, "bands merge below", ha="right", va="bottom", fontsize=8,
                color="#6B7689")
        ax.set_title(f"{TITLES[ds]}\nfit window: scan 0–{sv[keep][-1]:.2f}° "
                     f"({keep.sum()} of {len(sv)} steps)", fontsize=10)
        ax.set_xlabel("L2 phase scan value [deg]")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("bunch separation [µm]")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.text(0.5, 0.005, "green = L2 window kept for fitting (+ streak, energy-gated separation; "
             "the window ends where the curves stop agreeing or the bands merge)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


RUN_COLORS = {"E300_15671": "#eb6834", "TEST_15668": "#1baf7a", "E300_15673": "#2a78d6"}
RUN_SHORT = {"E300_15671": "15671 (deep, L1 ≈ −26°)", "TEST_15668": "15668 (middle, L1 ≈ −23.7°)",
             "E300_15673": "15673 (least, L1 ≈ −20°)"}


def figure_mismatch(res, path):
    """The July S2E's separation error against the compression state, all three setpoints on one
    axis: does the error follow the bunch separation, or the setpoint?"""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for ds, r in res.items():
        rp, sp = np.asarray(r["real_plus"]), np.asarray(r["sim_plus"])
        pre = np.arange(len(rp)) <= r["i_min"]
        keep = np.asarray(r["keep"])
        m = pre & np.isfinite(sp) & np.isfinite(rp)
        err = 100 * (sp - rp) / rp
        c = RUN_COLORS[ds]
        ax.plot(rp[m], err[m], "-", color=c, lw=1.0, alpha=0.6)
        ax.plot(rp[m & keep], err[m & keep], "o", color=c, ms=6, label=RUN_SHORT[ds])
        ax.plot(rp[m & ~keep], err[m & ~keep], "o", mfc="white", mec=c, ms=6)
    ax.axhspan(-15, 15, color="#8A93A6", alpha=0.12, lw=0)
    ax.axvline(FLOOR_UM, color="#6B7689", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("real bunch separation [µm]  (smaller = more compressed)")
    ax.set_ylabel("simulation − real [% of real]")
    ax.invert_xaxis()
    ax.legend(fontsize=8.5, loc="lower left", title="filled = in the fit window", title_fontsize=8)
    ax.set_title("FACET2-S2E separation error by setpoint (+ streak, before full compression)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel-tol", type=float, default=REL_TOL)
    ap.add_argument("--abs-tol", type=float, default=ABS_TOL_UM)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    res = {}
    for ds, (tag, l2ref) in RUNS.items():
        if not (S2R / "tier1_sim_scans" / ds / tag / "features.npz").exists():
            raise SystemExit(f"missing sim run {ds}/{tag}")
        r = select(ds, tag, l2ref, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
        res[ds] = r
        print(f"{ds}: L2_ref {r['l2ref_real_deg']:.2f} (delta + {r['delta_deg']['plus']:+.2f}, "
              f"- {r['delta_deg']['neg']:+.2f}), min at step {r['i_min']}, window {len(r['kept_steps'])} "
              f"steps 0-{r['kept_steps'][-1]}, window slope ratio sim/real {r['window_slope_ratio']}")
        for i, why in enumerate(r["reason"]):
            print(f"   {i:2d} {r['scan_vals_deg'][i]:4.2f}  real {r['real_plus'][i]:6.1f}  "
                  f"sim {r['sim_plus'][i]:6.1f}  slope {r['slope_ratio'][i]:5.2f}  "
                  f"neg real {r['real_neg'][i]:6.1f} sim {r['sim_neg'][i]:6.1f} "
                  f"{'ok' if r['neg_match'][i] else '--'}  {why}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"step_selection{args.tag}.json").write_text(json.dumps(res, indent=1))
    figure(res, OUT / "figures" / f"step_selection{args.tag}.png")
    figure_mismatch(res, OUT / "figures" / f"s2e_mismatch{args.tag}.png")


if __name__ == "__main__":
    main()
