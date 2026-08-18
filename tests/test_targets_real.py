"""D1 contracts: anchoring/crop geometry, flip commutation (the orientation
machinery relies on it), z-lobe split, fixed-pattern mask, and one real-shot
spot check against the Tier-0 record (skipped if the dataset is absent)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from xtcav_gpsr_tuning.screen import NX, NY, com_center_image  # noqa: E402
from xtcav_gpsr_tuning.targets_real import (  # noqa: E402
    XTCAV_ROOT, anchor_and_crop, camera_artifact_map, image_zlobe_split,
    repair_frame)


def _blob(H, W, cy, cx, sy=6.0, sx=10.0, amp=1.0):
    y, x = np.mgrid[0:H, 0:W]
    return amp * np.exp(-0.5 * ((y - cy) / sy) ** 2 - 0.5 * ((x - cx) / sx) ** 2)


def test_anchor_and_crop_geometry():
    # blob well off-center in the padded 300x450 hand-crop frame (rows >=184 zero)
    frame = np.zeros((300, 450))
    frame[:184] = _blob(184, 450, cy=60, cx=310)[:184]
    out = anchor_and_crop(frame)
    assert out.shape == (1, NY, NX)
    assert torch.isclose(out.sum(), torch.tensor(1.0, dtype=torch.float64))
    iy = torch.arange(NY, dtype=torch.float64)
    ix = torch.arange(NX, dtype=torch.float64)
    cy = float((out[0].sum(-1) * iy).sum() / out.sum())
    cx = float((out[0].sum(-2) * ix).sum() / out.sum())
    assert abs(cy - (NY - 1) / 2) < 0.05 and abs(cx - (NX - 1) / 2) < 0.05


def test_com_center_commutes_with_row_flip():
    img = torch.as_tensor(_blob(NY, NX, cy=50, cx=120)
                          + _blob(NY, NX, cy=100, cx=190, amp=0.4))[None]
    a = com_center_image(img.flip(-2))
    b = com_center_image(img).flip(-2)
    assert (a - b).abs().max() < 1e-9


def test_image_zlobe_split_two_blobs():
    img = torch.as_tensor(0.7 * _blob(NY, NX, cy=80, cx=100, sx=4.0)
                          + 0.3 * _blob(NY, NX, cy=80, cx=200, sx=4.0))[None]
    img = img / img.sum()
    hi = float(image_zlobe_split(img)[0])  # fraction on +x side of the COM
    assert 0.25 < hi < 0.35


def test_camera_artifact_map_flags_static_spike_not_jittering_beam():
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(40):
        # beam jitters shot-to-shot; artifact is static and point-like
        f = 200.0 * _blob(NY, NX, cy=80 + rng.normal(0, 4),
                          cx=150 + rng.normal(0, 8))
        f[40, 250] += 60.0
        frames.append(f)
    bad = camera_artifact_map(np.stack(frames))
    assert bad[40, 250]
    assert bad.sum() <= 25          # artifact + dilation only
    assert not bad[80, 150]         # beam core untouched
    rep = repair_frame(frames[0], bad)
    assert rep[40, 250] < frames[0][40, 250] - 30.0   # spike removed
    assert np.array_equal(rep[~bad], frames[0][~bad])  # everything else intact


@pytest.mark.skipif(not (XTCAV_ROOT / "E300_15673").exists(),
                    reason="E300_15673 raw data not present")
def test_real_shot_spot_check():
    from xtcav_gpsr_tuning.targets_real import load_dataset, processed_frame
    ds, cls, steps_per_shot, a, refs = load_dataset("E300_15673")
    assert len(cls["ix_plus"]) > 500 and len(cls["ix_neg"]) > 500
    ind = int(cls["ix_plus"][0])
    fr = processed_frame(a, ind, refs, int(steps_per_shot[ind]))
    assert fr.shape == (300, 450)         # zero-padded hand-crop frame
    assert fr[184:].max() == 0.0          # pad rows below the 184-px camera
    assert fr.sum() > 0
    out = anchor_and_crop(fr)
    assert float(out.sum()) == pytest.approx(1.0)
    # >90% of the anchored full-frame signal must survive the 300x160 crop
    full = com_center_image(torch.as_tensor(fr)[None]).clamp_min(0.0)
    H, W = full.shape[-2], full.shape[-1]
    r0, c0 = (H - NY) // 2, (W - NX) // 2
    kept = float(full[..., r0:r0 + NY, c0:c0 + NX].sum() / full.sum())
    assert kept > 0.9


def test_scale_bunch_per_bunch_semantics():
    from xtcav_gpsr_tuning.fit import scale_bunch
    rng = np.random.default_rng(3)
    cloud = torch.as_tensor(rng.normal(size=(4, 500, 6)))
    cloud[..., 5] += 10.0  # nonzero mean pz
    sd, ex = torch.tensor(np.log(2.0)), torch.tensor(np.log(1.5))
    out = scale_bunch(cloud, sd, ex)
    # means preserved per (batch, column)
    assert torch.allclose(out.mean(dim=-2), cloud.mean(dim=-2), atol=1e-12)
    # pz spread x2, x/px spread x1.5, y/py/z untouched
    r = out.std(dim=-2) / cloud.std(dim=-2)
    assert torch.allclose(r[..., 5], torch.tensor(2.0, dtype=r.dtype), atol=1e-9)
    assert torch.allclose(r[..., 0], torch.tensor(1.5, dtype=r.dtype), atol=1e-9)
    assert torch.allclose(r[..., 3], torch.tensor(1.5, dtype=r.dtype), atol=1e-9)
    for j in (1, 2, 4):
        assert torch.allclose(r[..., j], torch.tensor(1.0, dtype=r.dtype), atol=1e-9)
    # identity at zero log-scales
    zero = torch.tensor(0.0)
    assert torch.allclose(scale_bunch(cloud, zero, zero), cloud, atol=1e-12)


def test_no_quad_k1_leak_between_models():
    """Regression (2026-08-07): ForwardModel writes q.k1 onto the SHARED segment
    each forward; reading the baseline back off the segment made delta_tr compound
    across successive fits in one process."""
    import sys as _s
    from xtcav_gpsr_tuning import paths as _p
    _s.path.insert(0, str(_p.TWOBUNCH_SRC))
    from xtcav_gpsr_tuning.fit import ForwardModel
    from xtcav_gpsr_tuning.segment import element_by_name
    from xtcav_gpsr_tuning.stage_d import conditioning, load_stage_d

    cfg, setup = load_stage_d(device="cpu")
    base = conditioning(setup, cfg)[:1]
    pristine = float(element_by_name(setup.segment, "Q0FF").k1)

    m1 = ForwardModel(setup, base, cfg["voltage_V"], n_particles=64, seed=0,
                      anchor="none", image_anchor=True, base_off_norm=base)
    with torch.no_grad():
        m1.dtr[2] = 0.20
        m1(rows=[0])
    # a NEW model must not inherit m1's mutated quads
    m2 = ForwardModel(setup, base, cfg["voltage_V"], n_particles=64, seed=0,
                      anchor="none", image_anchor=True, base_off_norm=base)
    assert float(m2._k1_base[2]) == pytest.approx(pristine, rel=1e-12)
    with torch.no_grad():
        m2(rows=[0])
    assert float(element_by_name(setup.segment, "Q0FF").k1) == pytest.approx(
        pristine, rel=1e-12)
