"""Bmad ground truth -> Stage-C target images + per-sample TDC phase offsets.

Targets are rendered through the SAME `screen.render_image` code path as the fit
forward model (same-estimator principle). Phase offsets use the exact per-particle
regression on index-aligned valtrack beams (Stage-A-validated method) — the
image-centroid ± asymmetry version is kept only as the Stage-D-portable
cross-check. Under Stage C's translation-invariant loss the offset is inert
(pure x-translation to first order, ~4e-4 relative residual); it is measured and
recorded as convention bookkeeping (critic finding 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import paths
from .screen import ScreenGrid, render_image

paths.ensure_import_roots()
from pmd_beamphysics import ParticleGroup  # noqa: E402

C_LIGHT = 299792458.0
K_RF = 2 * np.pi * 11.424e9 / C_LIGHT


def load_dtotr_target(
    h5_path: str | Path,
    grid: ScreenGrid,
    anchor: str = "com",
    subsample: int | None = None,
    seed: int = 0,
) -> torch.Tensor:
    """One Bmad DTOTR beam -> (ny, nx) unit-sum target image.

    All particles (both weight classes) go through ONE render call so the charge
    split is preserved. `subsample` draws an independent macro-subset — the free
    noise-replica mechanism for the C4 statistics (critic finding 2)."""
    P = ParticleGroup(str(h5_path))
    idx = np.arange(len(P.x))
    if subsample is not None and subsample < len(idx):
        idx = np.random.default_rng(seed).choice(len(idx), subsample, replace=False)
    dev = grid.bins_x.device
    x = torch.tensor(P.x[idx], dtype=grid.dtype, device=dev)
    y = torch.tensor(P.y[idx], dtype=grid.dtype, device=dev)
    q = torch.tensor(P.weight[idx], dtype=grid.dtype, device=dev)
    with torch.no_grad():
        return render_image(x, y, q, grid, anchor=anchor)[0]


def target_charge_split(h5_path: str | Path) -> float:
    """Drive charge fraction from the weight tags (truth-level, for the loss term)."""
    P = ParticleGroup(str(h5_path))
    w = np.unique(P.weight)
    if len(w) < 2:
        return 1.0
    q_hi = P.weight[P.weight == w[-1]].sum()
    return float(q_hi / P.weight.sum())


def phase_offset_regression(sample_dir: str | Path, tag: str, voltage_V: float) -> dict:
    """Per-sample TDC phase offset via per-particle regression (exact method).

    Uses index-aligned (zero-loss) valtrack beams: Δx = x_DTOTR(±V) − x_DTOTR(off)
    regressed against z at XTCAVF exit. The kick is ∝ sin(2π φ0 ∓ k_rf z_eff + φ_off);
    at the zero crossing, intercept/slope = tan(φ_off)/k_rf — an R12-free estimator:

        φ_off = atan(k_rf · intercept / slope)

    Returns offsets from both streak signs (should agree; their spread is the
    internal systematic) in degrees.
    """
    d = Path(sample_dir)
    Px = ParticleGroup(str(d / f"{tag}_XTCAVF_off.h5"))
    P0 = ParticleGroup(str(d / f"{tag}_DTOTR_off.h5"))
    z = -C_LIGHT * (np.asarray(Px.t) - np.average(Px.t, weights=Px.weight))

    out = {}
    for case, sign in (("plus", +1.0), ("minus", -1.0)):
        f = d / f"{tag}_DTOTR_{case}.h5"
        if not f.exists():
            continue
        Pc = ParticleGroup(str(f))
        if len(Pc.x) != len(P0.x):
            out[case] = {"error": "particle count mismatch (losses) — estimator invalid"}
            continue
        dx = np.asarray(Pc.x) - np.asarray(P0.x)
        slope, intercept = np.polyfit(z, dx, 1)
        # Δx = R12·(±V/pc)·sin(k_rf·z + φ_off) ≈ a·z + b with a = R12(±V/pc)k_rf·cosφ,
        # b = R12(±V/pc)·sinφ → φ_off = atan(k_rf·b/a). R12, V, AND the voltage sign
        # all cancel in the ratio: plain atan, no branch folding (a sign flip negates
        # both a and b). Verified: plus/minus agree to <0.01° on valtracks.
        phi_rad = np.arctan(K_RF * intercept / slope)
        out[case] = {
            "phi_offset_deg": float(np.degrees(phi_rad)),
            "slope_m_per_m": float(slope),
            "intercept_um": float(intercept * 1e6),
        }
    if "plus" in out and "minus" in out and "error" not in out["plus"] | out["minus"]:
        out["phi_offset_deg_mean"] = 0.5 * (out["plus"]["phi_offset_deg"]
                                            + out["minus"]["phi_offset_deg"])
        out["phi_offset_deg_spread"] = abs(out["plus"]["phi_offset_deg"]
                                           - out["minus"]["phi_offset_deg"])
    return out


def phase_offsets_for_campaign(valtracks_dir: str | Path, voltage_V: float = 22e6) -> dict:
    """Run the regression estimator over every sample in a valtracks dir."""
    d = Path(valtracks_dir)
    manifest = json.loads((d / "valtracks_manifest.json").read_text())
    return {tag: phase_offset_regression(d, tag, voltage_V) for tag in sorted(manifest)}
