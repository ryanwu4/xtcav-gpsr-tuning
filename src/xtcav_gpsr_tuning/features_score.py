"""Stage-D2 physics-feature scoring (score only, never in the loss).

The controller's online XTCAV analysis (twobunch-s2e-rl ``online.features.image_features``:
valley-rule energy cut on the +V image, witness = lower-charge band, spacing from the
half-difference of the ±V band offsets, drive σz from the ±V drive x-variances, drive
charge fraction) applied IDENTICALLY to the real per-step target pairs and to the
model's rendered pairs, on the same 300x160 analysis grid and orientation.
"""

from __future__ import annotations

import numpy as np
import torch

from . import paths
from .screen import NX, NY, PIXEL_UM

paths.ensure_import_roots()

FEATURES = ("bunch_spacing", "drive_sigma_z", "drive_charge_fraction")


def grid_axes():
    x = (np.arange(NX) - (NX - 1) / 2.0) * PIXEL_UM * 1e-6
    y = (np.arange(NY) - (NY - 1) / 2.0) * PIXEL_UM * 1e-6
    return x, y


def streak_dxdz(setup, voltage_V: float) -> float:
    """Signed screen dx/dz of the as-built segment at +V (tau = -z, so dx/dz = -R[0][4])."""
    from .segment import element_by_name, numerical_rmatrix

    tdc = element_by_name(setup.segment, "XTCAVF")
    v0, p0 = tdc.voltage, tdc.phase
    dev = torch.device(setup.device)
    try:
        tdc.voltage = torch.tensor(float(voltage_V), dtype=torch.float64, device=dev)
        tdc.phase = torch.tensor(0.0, dtype=torch.float64, device=dev)
        R = numerical_rmatrix(setup.segment, setup.spec["meta"]["ref_energy_eV"], device=dev)
    finally:
        tdc.voltage, tdc.phase = v0, p0
    return -float(R[0][4])


def scan_features(imgs, streak: float) -> dict:
    """imgs: (2, B, ny, nx) [+V, -V] per-step images (tensor or array). Returns
    {feature: (B,) array (NaN where the valley rule finds no energy cut), 'valid': (B,)}."""
    from twobunch_s2e_rl.online.features import image_features

    a = imgs.detach().cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
    x, y = grid_axes()
    B = a.shape[1]
    out = {k: np.full(B, np.nan) for k in FEATURES}
    out["valid"] = np.zeros(B)
    for i in range(B):
        f = image_features(a[0, i], a[1, i], x, y, streak)
        out["valid"][i] = f.get("valid", 0.0)
        if f.get("peaks_ok"):
            for k in FEATURES:
                out[k][i] = f[k]
    return out


def feature_errors(model: dict, real: dict) -> dict:
    """Median |model - real| per feature over steps where both found a cut, plus counts."""
    res = {}
    for k in FEATURES:
        both = np.isfinite(model[k]) & np.isfinite(real[k])
        d = np.abs(model[k] - real[k])[both]
        res[k] = {"median_abs_err": float(np.median(d)) if d.size else float("nan"),
                  "n_both": int(both.sum())}
    return res
