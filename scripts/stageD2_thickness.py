"""Stage-D2 trace-thickness diagnostic: per-column energy-axis width of the drive and
witness bands, real vs the S2E renders of a probe (score_s2e_probe.py npz).

Each image is split at its own valley energy cut (online.features.valley_threshold on
the row projection); the witness is the minority-charge band. Width = intensity-weighted
mean over columns of the column's rms along the energy axis [um at 30.5 um/px]. The
2026-10-29 diagnostic found the real excess DRIVE-only (real 362-420 vs S2E ~105 um at
step 2); this asks which probe point (laser heater energy / timing, delay) closes it
without thickening the witness.

  PYTHONPATH=$PWD/src python scripts/stageD2_thickness.py --probe s2e_probe_15673_v3_heater_delay
-> data/<probe>/thickness_<conv>.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO.parent / "twobunch-s2e-rl" / "src"))
from stageD2_figures import targets  # noqa: E402
from twobunch_s2e_rl.online.features import valley_threshold  # noqa: E402

PX = 30.5


def band_width(img, band):
    """Mean column width [um] of the drive or witness band, NaN if no valley."""
    im = np.clip(img, 0, None)
    cut, ok = valley_threshold(im.sum(axis=1))
    if not ok:
        return np.nan
    below = np.arange(im.shape[0]) <= cut
    wit = below if im[below].sum() < im[~below].sum() else ~below
    im = im * (wit if band == "witness" else ~wit)[:, None]
    w = im.sum(0)
    cols = np.where(w > 0.004 * w.sum())[0]
    ys = np.arange(im.shape[0]) * PX
    W = []
    for c in cols:
        p = im[:, c] / w[c]
        m = (p * ys).sum()
        W.append(np.sqrt((p * (ys - m) ** 2).sum()))
    return float(np.average(W, weights=w[cols])) if len(W) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--conv", default="B")
    a = ap.parse_args()
    pdir = REPO / "data" / a.probe
    real = targets(a.conv)["images"].numpy()
    rows = json.loads((pdir / "manifest.json").read_text())
    steps = sorted({r["step"] for r in rows})
    out = {"real": {}, "points": {}}
    for s in steps:
        out["real"][s] = {k: {b: band_width(real[g, s], b) for b in ("drive", "witness")}
                          for g, k in enumerate(("plus", "minus"))}
    for r in rows:
        f = pdir / f"render_{a.conv}_{r['idx']:05d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        rec = {k: {b: band_width(z[k], b) for b in ("drive", "witness")} for k in ("plus", "minus")}
        rec["tv"] = {k: float(np.abs(z[k] - real[g, r["step"]]).sum())
                     for g, k in enumerate(("plus", "minus"))}
        out["points"].setdefault(r["injector_point"], {})[r["step"]] = rec
    (pdir / f"thickness_{a.conv}.json").write_text(json.dumps(out, indent=1))

    def per_step(d, b):  # mean over the two streak signs, per step
        return [np.nanmean([d[s][k][b] for k in ("plus", "minus")]) if s in d else np.nan
                for s in steps]

    hdr = " ".join(f"{s:>5d}" for s in steps)
    print(f"{'point':24s} drive width by step [um] {hdr} | witness {hdr} | TV vs real")
    for pt, d in [("real", out["real"])] + list(out["points"].items()):
        dw, ww = per_step(d, "drive"), per_step(d, "witness")
        tv = (f"{np.mean([d[s]['tv'][k] for s in d for k in ('plus', 'minus')]):.3f}"
              if pt != "real" else "")
        print(f"{pt:24s} {'':25s}{' '.join(f'{v:5.0f}' for v in dw)} |         "
              f"{' '.join(f'{v:5.0f}' for v in ww)} | {tv}")


if __name__ == "__main__":
    main()
