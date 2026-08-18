"""Jacobian SVD identifiability analysis (plan C3).

Computes ∂(image stack)/∂θ for θ = (17 knob-δ in σ_j units, 3 δ_tr in 1% k1
units, log ν_y, V_eff in MV) at the scan base point via CENTRAL FINITE
DIFFERENCES (2 forwards per parameter — CRN makes the forward deterministic, so
FD is exact to O(eps²); reverse-mode autograd would need one backward per image
pixel and forward-mode is unavailable through the flow's einsum path at scale).

SVDs the prior-scaled Jacobian and PRE-REGISTERS the identifiable subspace:
singular directions with s_i ≥ cut·s_max (default 0.05, recorded). The C4
Bmad-truth bias gate is evaluated ONLY on this subspace (critic finding 2).
"""

from __future__ import annotations

import json

import numpy as np
import torch


def _images_at(model, theta: np.ndarray) -> torch.Tensor:
    """Evaluate the forward at prior-scaled parameter vector theta (no grad).
    Order: [nd knob pulls, 3 dtr (1% k1), log nu_y, then — each only when the
    model actually fits it — veff (MV), log sdelta, log ex, l2_shift (deg)."""
    nd = len(model.s.fit_knobs)
    with torch.no_grad():
        delta_norm = torch.tensor(theta[:nd], dtype=torch.float64,
                                  device=model.base.device) * model.s.sigma_norm
        raw = torch.atanh((delta_norm / model.s.bound_norm).clamp(-0.99, 0.99))
        model.delta_raw.copy_(raw)
        model.dtr.copy_(torch.tensor(theta[nd:nd + 3] * 0.01, dtype=torch.float64,
                                     device=model.base.device))
        model.log_nu_y.copy_(torch.tensor(theta[nd + 3], dtype=torch.float64))
        j = nd + 4
        if model.veff_frac.requires_grad:
            model.veff_frac.copy_(torch.tensor(theta[j], dtype=torch.float64)
                                  * 1e6 / model.v_nominal)
            j += 1
        if model.log_sdelta.requires_grad:
            model.log_sdelta.copy_(torch.tensor(theta[j], dtype=torch.float64))
            j += 1
        if model.log_ex.requires_grad:
            model.log_ex.copy_(torch.tensor(theta[j], dtype=torch.float64))
            j += 1
        if model.l2_shift_deg.requires_grad:
            model.l2_shift_deg.copy_(torch.tensor(theta[j], dtype=torch.float64))
        imgs, _ = model()
    return imgs


def jacobian_svd(model, cut: float = 0.05, downsample: int = 2,
                 eps: float = 0.25) -> dict:
    # Only parameters the fit actually varies (audit #7): V_eff is frozen in
    # Stage D (fit_veff=False), so including it put 40-44% of two kept
    # directions' norm on a coordinate that is identically zero; l2_shift_deg
    # IS fitted and was missing from the accounting entirely.
    names = ([f"delta:{k}" for k in model.s.fit_knobs]
             + [f"dtr:{n}" for n in model.s.dtr_names]
             + ["nu_y(log)"]
             + (["veff(MV)"] if model.veff_frac.requires_grad else [])
             + (["sdelta(ln)"] if model.log_sdelta.requires_grad else [])
             + (["ex(ln)"] if model.log_ex.requires_grad else [])
             + (["l2_shift(deg)"] if model.l2_shift_deg.requires_grad else []))
    n_par = len(names)
    ds = downsample

    cols = []
    for j in range(n_par):
        tp = np.zeros(n_par); tp[j] = +eps
        tm = np.zeros(n_par); tm[j] = -eps
        ip = _images_at(model, tp)[..., ::ds, ::ds]
        im = _images_at(model, tm)[..., ::ds, ::ds]
        cols.append(((ip - im) / (2 * eps)).reshape(-1).cpu().numpy())
    _images_at(model, np.zeros(n_par))  # restore base point
    J = np.stack(cols, axis=1)

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    keep = S >= cut * S[0]
    leverage = np.sqrt((Vt[keep] ** 2 * S[keep, None] ** 2).sum(0))
    leverage = leverage / leverage.max()
    return {
        "param_names": names,
        "singular_values": S.tolist(),
        "cut": cut,
        "eps": eps,
        "n_kept": int(keep.sum()),
        "kept_directions": Vt[keep].tolist(),
        "leverage": dict(zip(names, map(float, leverage))),
        "note": "delta in sigma_j units, dtr in 1% k1 units, veff in MV — "
                "prior-scaled coordinates; central FD under CRN; leverage = rms "
                "row of kept V^T weighted by singular values, normalized to max",
    }


def save(result: dict, path):
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
