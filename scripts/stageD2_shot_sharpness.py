"""Stage-D2: is the real trace thickness in single shots, or does it come from averaging them?

Rebuilds the convention-B targets shot by shot (targets_real.build_targets selection, clean
row-distortion references as in build_targets_variants.py --variant clean) and, per step and
streak sign, compares:
  - the drive / witness band widths along the energy axis (stageD2_thickness.band_width) of each
    anchored single shot vs those of the per-step mean (= the fit target);
  - a cross-correlation-aligned mean: every shot re-shifted (integer px, both axes) onto the COM
    mean before averaging; its width and its TV to the COM mean bound how much of the target's
    blur is shot-to-shot misalignment that COM centring leaves in;
  - the witness band's width along the TIME axis (columns), single shots vs the mean;
  - raw camera saturation: per shot, the fraction of pixels at the frame's uint16 ceiling.
The COM and aligned means are also saved in the targets layout (canonical frame, (step, sign)),
so a fit or a blur bound can be run against the aligned version.

  PYTHONPATH=$PWD/src MPLBACKEND=Agg python scripts/stageD2_shot_sharpness.py
-> results/stageD2/15673/shot_sharpness.json, shot_means_B.npz (+ a printed table)
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from build_targets_variants import clean_refs  # noqa: E402
from stageD2_thickness import band_width  # noqa: E402
from xtcav_gpsr_tuning import targets_real as tr  # noqa: E402

RAW = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/E300_15673/images/DTOTR2")


def xcorr_align(shots, ref, max_shift=12):
    """Integer-pixel shift of each shot that maximises its overlap with ref (FFT correlation)."""
    F = np.fft.rfft2(ref)
    out = []
    for s in shots:
        c = np.fft.irfft2(np.fft.rfft2(s) * np.conj(F), s=s.shape)
        c = np.fft.fftshift(c)
        cy, cx = np.array(c.shape) // 2
        win = c[cy - max_shift:cy + max_shift + 1, cx - max_shift:cx + max_shift + 1]
        dy, dx = np.unravel_index(np.argmax(win), win.shape)
        out.append(np.roll(s, (max_shift - dy, max_shift - dx), axis=(0, 1)))
    return np.stack(out)


def witness_time_width(img):
    """rms width [um] along the time axis (columns) of the witness band (minority side of the valley)."""
    from twobunch_s2e_rl.online.features import valley_threshold
    im = np.clip(img, 0, None)
    cut, ok = valley_threshold(im.sum(axis=1))
    if not ok:
        return np.nan
    below = np.arange(im.shape[0]) <= cut
    wit = below if im[below].sum() < im[~below].sum() else ~below
    prof = (im * wit[:, None]).sum(0)
    xs = np.arange(im.shape[1]) * 30.5
    prof = prof / prof.sum()
    m = (prof * xs).sum()
    return float(np.sqrt((prof * (xs - m) ** 2).sum()))


def saturation():
    """Per step file: fraction of shots with any pixel at 65535 / at the step's own maximum."""
    rec = {}
    for f in sorted(RAW.glob("DTOTR2_data_step*.h5")):
        with h5py.File(f, "r") as h:
            d = h["entry/data/data"][()]
        mx = d.reshape(len(d), -1).max(axis=1)
        top = int(mx.max())
        rec[f.stem[-2:]] = {"frame_max_median": float(np.median(mx)), "frame_max_top": top,
                            "frac_shots_at_65535": float((mx == 65535).mean()),
                            "frac_shots_within_1pct_of_top": float((mx >= 0.99 * top).mean()),
                            "px_at_top_per_shot": float(np.mean([(x == top).sum() for x in d]))}
    return rec


def main():
    ds, cls, sps, a, _ = tr.load_dataset("E300_15673")
    refs = clean_refs(a, cls, sps)
    steps = np.unique(sps[np.asarray(cls["ix_plus"], int)])
    steps = steps[steps >= 0]
    out = {"steps": {}, "saturation": saturation()}
    NYX = (len(steps), 2, tr.NY, tr.NX)
    com_means, al_means = np.zeros(NYX), np.zeros(NYX)
    for si, step in enumerate(steps):
        rec = {}
        for gi, sign in enumerate(("plus", "neg")):
            keep, _, _ = tr.select_shots(cls, sps, step, sign, 0.2, 0.3, 5)
            fr = [tr.processed_frame(a, int(i), refs, int(sps[i])) for i in keep]
            fr = [f for f in fr if f.sum() > 0.2 * 5.05e5]
            if len(fr) < 3:
                continue
            shots = tr.anchor_and_crop(np.stack(fr)).numpy()
            mean = shots.mean(0)
            mean /= mean.sum()
            al = xcorr_align(shots, mean)
            amean = al.mean(0)
            amean /= amean.sum()
            com_means[si, gi], al_means[si, gi] = mean, amean
            w = lambda im, b: band_width(im, b)
            sd = np.array([w(s, "drive") for s in shots])
            sw = np.array([w(s, "witness") for s in shots])
            rec[sign] = {"n": len(shots),
                         "drive_shot_median_um": float(np.nanmedian(sd)),
                         "drive_mean_um": w(mean, "drive"), "drive_aligned_mean_um": w(amean, "drive"),
                         "witness_shot_median_um": float(np.nanmedian(sw)),
                         "witness_mean_um": w(mean, "witness"),
                         "witness_aligned_mean_um": w(amean, "witness"),
                         "witness_time_shot_median_um": float(np.nanmedian([witness_time_width(x) for x in shots])),
                         "witness_time_mean_um": witness_time_width(mean),
                         "witness_time_aligned_mean_um": witness_time_width(amean),
                         "tv_aligned_vs_com_mean": float(np.abs(amean - mean).sum()),
                         "tv_shot_vs_mean_median": float(np.median([np.abs(s - mean).sum() for s in shots]))}
        out["steps"][int(step)] = {"scan_deg": float(ds.params["scanVals"][si]), **rec}
        p = rec.get("plus", {})
        if p:
            print(f"step {step:2d} scan {ds.params['scanVals'][si]:+.2f}  n {p['n']:2d}  drive shot/mean/aligned "
                  f"{p['drive_shot_median_um']:5.0f}/{p['drive_mean_um']:5.0f}/{p['drive_aligned_mean_um']:5.0f} um  "
                  f"witness {p['witness_shot_median_um']:4.0f}/{p['witness_mean_um']:4.0f}/{p['witness_aligned_mean_um']:4.0f}  "
                  f"witness-time {p['witness_time_shot_median_um']:4.0f}/{p['witness_time_mean_um']:4.0f}/{p['witness_time_aligned_mean_um']:4.0f}  "
                  f"TV aligned-vs-COM mean {p['tv_aligned_vs_com_mean']:.3f}  shot-vs-mean {p['tv_shot_vs_mean_median']:.3f}",
                  flush=True)
    np.savez_compressed(REPO / "results/stageD2/15673/shot_means_B.npz", com=com_means, aligned=al_means,
                        steps=steps)
    path = REPO / "results/stageD2/15673/shot_sharpness.json"
    path.write_text(json.dumps(out, indent=1))
    s = out["saturation"]
    print("saturation: frames with a pixel at 65535 in", sum(v["frac_shots_at_65535"] > 0 for v in s.values()),
          "of", len(s), "step files; median frame max", np.median([v["frame_max_median"] for v in s.values()]))
    print(path)


if __name__ == "__main__":
    main()
