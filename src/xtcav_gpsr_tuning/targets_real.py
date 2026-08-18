"""Stage-D real DTOTR2 target construction (plan D1) — dataset-agnostic core.

Reuses the deployed E300 analysis pipeline verbatim (yiheng's xtcav.py +
DAQ_analysis_class, the Tier-0 import pattern) up to but NOT including the
deployed auto-crop: background subtraction → hand crop (zero-padded 450x300
frame) → median/hot-pixel/threshold → per-step off-shot row-distortion
correction. Anchoring then happens in IMAGE space with the torch op shared
verbatim with the fit's forward model (`screen.com_center_image` — critic M6):
each processed frame is COM-centered sub-pixel in the 450x300 hand-crop frame,
center-cropped to the 300x160 analysis grid (30.5 µm/px, asserted against the
camera RESOLUTION metadata), and unit-sum normalized. Per-(step,sign) targets
are means over the anchored on-shots within ±phase_window of the step's median
TCAV phase; per-target noise floors come from split-half means.

Orientation: targets are stored in the CANONICAL camera frame (rows = camera y
= energy axis, plus = ix_plus). The fit's 4 discrete orientation/pairing combos
are generated downstream by flipping rows (COM-centering commutes with the flip
about the even-size frame center) and/or swapping the sign axis.

Charge-split targets use the z-lobe split (never the energy gate): the soft
sigmoid split about the anchored image's x-COM, the image-side analog of
`screen.charge_split` (same 30 µm ≈ 1 px temperature).

Runs under slac-rl (the DAQ class stack imports there; torch required for the
shared anchor op). No GPU needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import paths
from .screen import NX, NY, PIXEL_UM, com_center_image, image_zlobe_split

XTCAV_ROOT = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
sys.path.insert(0, str(XTCAV_ROOT / "yiheng" / "DAQ_analysis_class"))
sys.path.insert(0, str(XTCAV_ROOT / "yiheng" / "XTCAV_analysis_script"))

# Notebook-tuned preprocessing constants, identical to Tier-0
# (twobunch-s2e-rl/scripts/xtcav_real_features.py — kept in sync by
# tests/test_targets_real.py).
HAND_CROP = (200, 0, 650, 300)
HOT_PIX_THRESHOLD = 1e3
NOISE_THRESHOLD = 35
MEDIAN_FILTER = 3
D_NOM = 59.31  # mm

SPLIT_TEMP_M = 30e-6  # z-lobe soft-split temperature, ~1 px (screen.charge_split)


def _import_deployed():
    import matplotlib
    matplotlib.use("Agg")
    import xtcav  # noqa: F401
    from dataset import DATASET  # noqa: F401
    return xtcav, DATASET


def load_dataset(name: str):
    """DATASET + shot classification + per-shot steps + XTCAVAnalysis (auto_crop
    DISABLED — the anchor is ours) + per-step row-distortion references."""
    xtcav, DATASET = _import_deployed()
    experiment, number = name.rsplit("_", 1)
    ds = DATASET(str(XTCAV_ROOT), experiment, number)
    cls = xtcav.classify_tcav_shots(ds)
    steps_per_shot = np.asarray(xtcav.get_shot_steps(ds))
    a = xtcav.XTCAVAnalysis(
        ds, cam="DTOTR2", d_nom=D_NOM, chargeCal=1.0,
        hand_crop=HAND_CROP, auto_crop=None,
        hot_pix_threshold=HOT_PIX_THRESHOLD, noise_threshold=NOISE_THRESHOLD,
        median_filter=MEDIAN_FILTER,
    )
    refs = xtcav.build_step_distortion_references(a, cls,
                                                  steps_per_shot=steps_per_shot,
                                                  verbose=False)
    return ds, cls, steps_per_shot, a, refs


def processed_frame(a, ind: int, refs, step: int) -> np.ndarray:
    """One shot through the deployed preprocessing + row-shift correction,
    in the zero-padded hand-crop frame (float, raw counts)."""
    import xtcav
    img = a.get_image(int(ind))
    assert abs(img.cal * 1e3 - PIXEL_UM) < 1e-6, \
        f"camera cal {img.cal} mm/px != grid {PIXEL_UM} um/px"
    ref = refs.get(int(step), refs.get(None))
    if ref is not None and img.data.shape[0] == len(ref["distortion_px"]):
        sh = xtcav.compute_distortion_shifts(ref["off_avg"], ref["distortion_px"])
        shifts = np.where(np.isfinite(sh["shifts_px"]), sh["shifts_px"], 0.0)
        img.data = xtcav.apply_row_shift(img.data, shifts)
    return np.asarray(img.data, dtype=np.float64)


def anchor_and_crop(frames: np.ndarray) -> torch.Tensor:
    """(k, H, W) processed hand-crop frames -> (k, NY, NX) anchored unit-sum
    images: COM-center in the FULL frame (shared torch op), center-crop to the
    analysis grid, renormalize. Frames with no signal raise."""
    t = torch.as_tensor(np.ascontiguousarray(frames), dtype=torch.float64)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    tot = t.sum(dim=(-2, -1))
    if (tot <= 0).any():
        raise ValueError("empty frame passed to anchor_and_crop")
    t = com_center_image(t)
    H, W = t.shape[-2], t.shape[-1]
    r0, c0 = (H - NY) // 2, (W - NX) // 2
    t = t[..., r0:r0 + NY, c0:c0 + NX].clamp_min(0.0)
    return t / t.sum(dim=(-2, -1), keepdim=True)


# image_zlobe_split now lives in screen.py so the MODEL uses the same
# estimator (audit #5); re-exported here for the existing call sites/tests.


def _tv(a: torch.Tensor, b: torch.Tensor) -> float:
    """Per-image-pair TV distance, same convention as the fit's image loss."""
    return float((a - b).abs().sum(dim=(-2, -1)).mean())


def select_shots(cls, steps_per_shot, step: int, sign: str,
                 phase_window_deg: float, widen_deg: float,
                 min_shots: int = 5):
    """On-shot indices for (step, sign) within ±window of the step's median
    TCAV phase; widen once if fewer than min_shots survive. Returns
    (kept_indices, median_phase, window_used)."""
    ix = np.asarray(cls["ix_plus"] if sign == "plus" else cls["ix_neg"], int)
    ix = ix[steps_per_shot[ix] == step]
    if ix.size == 0:
        return ix, np.nan, phase_window_deg
    ph = np.asarray(cls["phase"], float)[ix]
    med = float(np.median(ph))
    for win in (phase_window_deg, widen_deg):
        keep = ix[np.abs(ph - med) <= win]
        if keep.size >= min_shots:
            return keep, med, win
    return keep, med, widen_deg  # whatever survived at the widest window


def camera_artifact_map(frames: np.ndarray, ratio: float = 3.0,
                        abs_floor: float = 10.0, patch: int = 1) -> np.ndarray:
    """Detect fixed-pattern camera artifacts (the documented hot-pixel blemish
    at z≈−78 µm, ΔE/E≈−0.8%, plus neighbors) in the CAMERA frame, where they
    are actually fixed: the per-pixel MEDIAN over many shots kills the jittering
    beam but keeps static structure; flagged = median exceeds its own 5x5 local
    median by `ratio`× plus `abs_floor` counts (point-like AND persistent).
    Measured on E300_15673: 3 clusters, rows 127–136, cols 214–236, ~40 px.
    COM anchoring moves the camera frame per shot, so these must be REPAIRED at
    source (repair_frame), not masked in the anchored grid where they smear.
    Returns bool (H,W), True = artifact."""
    from scipy.ndimage import binary_dilation, median_filter
    med = np.median(frames, axis=0)
    smooth = median_filter(med, size=5)
    bad = med > ratio * smooth + abs_floor
    if patch > 0:
        bad = binary_dilation(bad, iterations=patch)
    return bad


def repair_frame(frame: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Replace artifact pixels with their 5x5 local median (the same op the
    deployed hot-pixel filter uses for above-threshold pixels)."""
    from scipy.ndimage import median_filter
    out = frame.copy()
    out[bad] = median_filter(frame, size=5)[bad]
    return out


def build_targets(name: str, phase_window_deg: float = 0.2,
                  widen_deg: float = 0.3, off_pool_halfwidth: int = 2,
                  min_shots: int = 5, off_amp_max: float = 1.0,
                  min_counts_frac: float = 0.2, verbose: bool = True):
    """Full D1 product for one dataset. Returns a dict of arrays (npz-ready)."""
    ds, cls, steps_per_shot, a, refs = load_dataset(name)
    # TCAV-OFF AMPLITUDE GATE (audit 2026-08-07, finding #2). The deployed
    # classifier defines off as amp < mean_on_amp - 0.5, which on this dataset
    # is amp < 21.51 and admits 348 of 400 "off" shots at 19.4-21.5, i.e.
    # 88-98% of FULL streak amplitude. Averaged into the off targets those
    # produce streaked content the V=0 forward can never render (18 of 33
    # targets were 20-27% contaminated; stored off x-rms was bimodal,
    # 117-145 um clean vs 426-1120 um contaminated). Keep only true zeros.
    amp_all = np.asarray(cls["amp"], float)
    ix_off_all = np.asarray(cls["ix_off"], int)
    ix_off_clean = ix_off_all[amp_all[ix_off_all] < off_amp_max]
    if verbose:
        print(f"  off-shot gate (amp < {off_amp_max}): {ix_off_clean.size}/"
              f"{ix_off_all.size} kept (mean_on_amp {cls['mean_on_amp']:.2f})",
              flush=True)
    if ix_off_clean.size == 0:
        raise RuntimeError("off-shot amplitude gate removed every off shot")
    steps = np.unique(steps_per_shot[np.asarray(cls["ix_plus"], int)])
    steps = steps[steps >= 0]
    n_steps = len(steps)
    scan_vals = np.asarray(ds.params["scanVals"], float)

    imgs = np.zeros((n_steps, 2, NY, NX))
    floors = np.full((n_steps, 2), np.nan)          # split-half TV floor of the mean
    splits = np.full((n_steps, 2), np.nan)          # z-lobe split of the target
    split_iqr = np.full((n_steps, 2), np.nan)       # per-shot split IQR
    n_used = np.zeros((n_steps, 2), int)
    med_phase = np.full((n_steps, 2), np.nan)
    win_used = np.full((n_steps, 2), np.nan)
    off_imgs = np.zeros((n_steps, NY, NX))
    off_n = np.zeros(n_steps, int)
    off_floors = np.full(n_steps, np.nan)
    off_xrms = np.full(n_steps, np.nan)

    cache: dict[int, np.ndarray] = {}
    n_empty = 0

    def raw_frame(i, step):
        if i not in cache:
            cache[int(i)] = processed_frame(a, int(i), refs, int(step))
        return cache[int(i)]

    # ---- fixed-pattern artifact map from a seeded subsample (camera frame) ----
    rng = np.random.default_rng(0)
    sub = np.concatenate([
        rng.choice(cls["ix_plus"], min(60, len(cls["ix_plus"])), replace=False),
        rng.choice(cls["ix_neg"], min(60, len(cls["ix_neg"])), replace=False),
        rng.choice(cls["ix_off"], min(40, len(cls["ix_off"])), replace=False)])
    stack = [raw_frame(int(i), steps_per_shot[i]) for i in sub]
    bad_map = camera_artifact_map(np.stack([f for f in stack if f.sum() > 0]))
    # DEAD-FRAME FLOOR. Unit-sum normalization turns a no-beam frame into pure
    # diffuse noise, which then dominates any small pool it lands in. E300_15673
    # has exactly one such off shot (index 692, 146 counts vs a ~4.5e5 median);
    # with the off pool half-width of 2 it alone corrupted steps 3-7 (pooled
    # x-rms 760-840 um against a true off spot of ~120 um). Reject anything
    # below min_counts_frac of the dataset median.
    ref_counts = float(np.median([f.sum() for f in stack if f.sum() > 0]))
    counts_floor = min_counts_frac * ref_counts
    if verbose:
        print(f"  dead-frame floor: {counts_floor:.3g} counts "
              f"({min_counts_frac:g} x median {ref_counts:.3g})", flush=True)

    def frames(inds, steps_of):
        """Repaired processed frames for `inds`, dropping empty (all-zero after
        thresholding) shots. Returns (k,H,W) or None."""
        nonlocal n_empty
        out = []
        for i in inds:
            f = raw_frame(int(i), steps_of[i])
            if f.sum() > counts_floor:
                out.append(repair_frame(f, bad_map))
            else:
                n_empty += 1
        return np.stack(out) if out else None

    for si, step in enumerate(steps):
        for gi, sign in enumerate(("plus", "neg")):
            keep, med, win = select_shots(cls, steps_per_shot, step, sign,
                                          phase_window_deg, widen_deg,
                                          min_shots)
            med_phase[si, gi], win_used[si, gi] = med, win
            fr = frames(keep, steps_per_shot) if keep.size else None
            n_used[si, gi] = 0 if fr is None else fr.shape[0]
            if fr is None:
                continue
            anchored = anchor_and_crop(fr)
            mean = anchored.mean(dim=0)
            imgs[si, gi] = (mean / mean.sum()).numpy()
            shot_splits = image_zlobe_split(anchored).numpy()
            splits[si, gi] = float(image_zlobe_split(mean.unsqueeze(0))[0])
            if anchored.shape[0] >= 2:
                q75, q25 = np.percentile(shot_splits, [75, 25])
                split_iqr[si, gi] = q75 - q25
                ha = anchored[0::2].mean(dim=0)
                hb = anchored[1::2].mean(dim=0)
                floors[si, gi] = 0.5 * _tv(ha / ha.sum(), hb / hb.sum())
        # TCAV-off channel: pool over adjacent steps (the amplitude gate leaves
        # only ~1.6 true-off shots per step, so the pool half-width is wider
        # than the pre-gate version while still resolving the off-spot trend)
        ix_off = ix_off_clean
        lo, hi = step - off_pool_halfwidth, step + off_pool_halfwidth
        pool = ix_off[(steps_per_shot[ix_off] >= lo)
                      & (steps_per_shot[ix_off] <= hi)]
        fr_off = frames(pool, steps_per_shot) if pool.size else None
        off_n[si] = 0 if fr_off is None else fr_off.shape[0]
        if fr_off is not None:
            anchored = anchor_and_crop(fr_off)
            m = anchored.mean(dim=0)
            off_imgs[si] = (m / m.sum()).numpy()
            if anchored.shape[0] >= 2:   # split-half floor, as for the on-shots
                ha, hb = anchored[0::2].mean(dim=0), anchored[1::2].mean(dim=0)
                off_floors[si] = 0.5 * _tv(ha / ha.sum(), hb / hb.sum())
            xs = np.arange(NX) * PIXEL_UM
            prof = off_imgs[si].sum(axis=-2); prof = prof / prof.sum()
            off_xrms[si] = float(np.sqrt((prof * (xs - (prof * xs).sum()) ** 2).sum()))
        if verbose:
            print(f"  step {step:3d} (scan {scan_vals[si]:+.2f} deg): "
                  f"n+={n_used[si, 0]} n-={n_used[si, 1]} off={off_n[si]} "
                  f"floors {floors[si, 0]:.4f}/{floors[si, 1]:.4f} "
                  f"off_xrms {off_xrms[si]:.0f}um", flush=True)

    # Artifacts are repaired at source (camera frame), so no loss masking is
    # needed; the all-ones mask keeps the loss code path uniform.
    mask = np.ones((NY, NX))
    ys, xs = np.where(bad_map)

    provenance = {
        "dataset": name, "hand_crop": HAND_CROP,
        "hot_pix_threshold": HOT_PIX_THRESHOLD,
        "noise_threshold": NOISE_THRESHOLD, "median_filter": MEDIAN_FILTER,
        "pixel_um": PIXEL_UM, "grid": [NY, NX],
        "phase_window_deg": phase_window_deg, "widen_deg": widen_deg,
        "min_shots": min_shots, "off_pool_halfwidth": off_pool_halfwidth,
        "off_amp_max": off_amp_max,
        "n_off_shots_gated_out": int(ix_off_all.size - ix_off_clean.size),
        "anchor": "image_com (screen.com_center_image, shared with the fit)",
        "orientation": "canonical camera frame; rows=y/energy, plus=ix_plus",
        "row_distortion_correction": "per-step off-shot references (deployed)",
        "split_temp_m": SPLIT_TEMP_M,
        "artifact_repair": "camera-frame fixed-pattern pixels replaced by 5x5 "
                           "local median before anchoring (mask kept all-ones)",
        "n_artifact_px": int(bad_map.sum()),
        "artifact_bbox_rows": [int(ys.min()), int(ys.max())] if len(ys) else None,
        "artifact_bbox_cols": [int(xs.min()), int(xs.max())] if len(ys) else None,
        "n_lowsignal_shots_dropped": n_empty,
        "min_counts_frac": min_counts_frac,
        "dead_frame_counts_floor": counts_floor,
    }
    return {
        "images": imgs, "off_images": off_imgs, "mask": mask,
        "camera_artifact_map": bad_map,
        "splits": splits, "split_iqr": split_iqr, "floors_tv": floors,
        "off_floors_tv": off_floors, "off_xrms_um": off_xrms,
        "n_shots": n_used, "n_off": off_n, "steps": steps,
        "scan_vals_deg": scan_vals[:n_steps], "median_phase_deg": med_phase,
        "phase_window_used_deg": win_used,
        "provenance": np.array(json.dumps(provenance)),
    }
