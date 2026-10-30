"""Stage-D2 S2E probe: render saved FACET2-S2E XTCAVF beams (Impact-T + Bmad, no
surrogate) through the SAME fit segment + screen + anchoring as a fitted
ForwardModel, with that model's calibration state (ν_y, δ_tr, V_eff, row
correction). Used to ask whether out-of-box injector moves bring the images
closer to the real targets in the simulation itself, before retraining the CNF.

Saved S2E beams (getBeamAtElement) are at the XTCAVF EXIT with z = -c (t - <t>_w)
already, i.e. the flow's own training frame (convert.cloud_to_beam docstring), so
the same back-drift and tau = -z apply. Drive = the higher of the two particle
weights (FACET2-S2E tag); per-particle charge = the openPMD weight.
"""

from __future__ import annotations

import numpy as np
import torch

from .convert import cloud_to_beam
from .fit import ForwardModel, off_row_shifts, shift_rows
from .screen import render_image
from .segment import back_drift

COORDS = ("x", "y", "z", "px", "py", "pz")


def load_s2e_cloud(path):
    """(cloud (N,6) float64, charges (N,) C, n_drive) with drive particles first."""
    from pmd_beamphysics import ParticleGroup

    P = ParticleGroup(str(path))
    w = np.asarray(P.weight, float)
    drive = w == w.max()
    cloud = np.stack([np.asarray(getattr(P, k), float) for k in COORDS], axis=1)
    order = np.concatenate([np.where(drive)[0], np.where(~drive)[0]])
    return cloud[order], w[order], int(drive.sum())


@torch.no_grad()
def render_s2e(m: ForwardModel, path) -> dict:
    """+V / −V / off unit-sum images of one saved S2E beam under model m's
    calibration state, anchored exactly as m's own renders."""
    cloud, q, n_drive = load_s2e_cloud(path)
    dev = m.base.device
    c = torch.as_tensor(cloud, dtype=torch.float64, device=dev).unsqueeze(0)
    qt = torch.as_tensor(q, dtype=torch.float64, device=dev).unsqueeze(0)
    for i, (quad, k1b) in enumerate(zip(m._quads, m._k1_base)):
        quad.k1 = k1b * (1.0 + m.dtr[i])
    beam = back_drift(cloud_to_beam(c, m.E_ref, qt, device=dev), m.handoff_back_drift_m)
    raw = []
    for sign in (+1.0, -1.0, 0.0):
        m._tdc.voltage = sign * m.v_nominal * (1.0 + m.veff_frac)
        m._tdc.phase = torch.tensor(m.phi, dtype=torch.float64, device=dev)
        out = m.s.segment.track(beam)
        raw.append(render_image(out.x, out.y, qt, m.s.grid, anchor="none",
                                y_scale=torch.exp(m.log_nu_y)))
    if m.row_center:
        sh = off_row_shifts(raw[2])
        raw = [shift_rows(r, sh) for r in raw]
    imgs = [m._anchor(r)[0].cpu().numpy() for r in raw]
    return {"plus": imgs[0], "minus": imgs[1], "off": imgs[2], "n_drive": n_drive,
            "n_total": int(len(q)), "q_drive_nC": float(q[:n_drive].sum() * 1e9),
            "q_witness_nC": float(q[n_drive:].sum() * 1e9)}
