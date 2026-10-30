"""Stage-D2 pieces that need no flow: the model-side row correction must be the
deployed op (values) with a correct first-order response (gradients)."""

import numpy as np
import pytest
import torch

from xtcav_gpsr_tuning.fit import (_interp_matrix, _row_fits, off_row_shifts,
                                   shift_rows)


def _deployed():
    from xtcav_gpsr_tuning.targets_real import _import_deployed

    xtcav, _ = _import_deployed()
    import analysis as an
    return xtcav, an


def _tilted_spot(ny=40, nx=60, tilt=0.15, x0=28.3, sig=3.0, ybar=20.0, sy=6.0):
    y = np.arange(ny)[:, None]
    x = np.arange(nx)[None, :]
    c = x0 + tilt * (y - ybar)
    return np.exp(-0.5 * ((x - c) / sig) ** 2) * np.exp(-0.5 * ((y - ybar) / sy) ** 2)


def test_shift_rows_matches_apply_row_shift():
    xtcav, _ = _deployed()
    rng = np.random.default_rng(0)
    img = rng.random((12, 30))
    sh = rng.uniform(-4, 4, 12)
    ref = xtcav.apply_row_shift(img, sh)
    got = shift_rows(torch.tensor(img), torch.tensor(sh)).numpy()
    # interior columns agree exactly; only the frame-edge column can differ (scipy cval rule)
    np.testing.assert_allclose(got[:, 5:-5], ref[:, 5:-5], atol=1e-12)


def test_row_fits_parity():
    xtcav, an = _deployed()
    im = _tilted_spot()
    a = xtcav.fit_row_distortion(im)
    b, popts = _row_fits(im, an)
    assert np.array_equal(np.isnan(a), np.isnan(b))
    np.testing.assert_array_equal(a[np.isfinite(a)], b[np.isfinite(b)])
    assert sorted(popts) == list(np.where(np.isfinite(b))[0])


def test_interp_matrix_reproduces_np_interp():
    rng = np.random.default_rng(1)
    finite = rng.random(25) > 0.4
    finite[[0, -1]] = False
    x0 = rng.normal(size=25)
    idx = np.where(finite)[0]
    ref = np.interp(np.arange(25), idx, x0[idx])
    np.testing.assert_allclose(_interp_matrix(finite)[:, idx] @ x0[idx], ref, atol=1e-12)


def test_off_row_shift_values_and_gradient():
    xtcav, _ = _deployed()
    im = torch.tensor(_tilted_spot(), requires_grad=True)
    sh = off_row_shifts(im.unsqueeze(0))[0]
    d = xtcav.fit_row_distortion(im.detach().numpy())
    ref = xtcav.compute_distortion_shifts(im.detach().numpy(), d)["shifts_px"]
    np.testing.assert_allclose(sh.detach().numpy(), np.where(np.isfinite(ref), ref, 0.0))
    # directional derivative along a change of tilt vs central finite differences
    v = torch.tensor(_tilted_spot(tilt=0.16) - _tilted_spot(tilt=0.14)) / 0.02
    e = torch.zeros((), dtype=torch.float64, requires_grad=True)
    s_e = off_row_shifts((im.detach() + e * v).unsqueeze(0))[0]
    rows = [5, 12, 20, 28, 34]
    ad = np.array([float(torch.autograd.grad(s_e[r], e, retain_graph=True)[0]) for r in rows])
    h = 1e-3
    fd = ((off_row_shifts((im.detach() + h * v).unsqueeze(0))[0]
           - off_row_shifts((im.detach() - h * v).unsqueeze(0))[0]) / (2 * h)).numpy()[rows]
    np.testing.assert_allclose(ad, fd, rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize("u0", [0.5, 0.643])
def test_asymmetric_bound_spans_the_box(u0):
    lo, hi = torch.tensor([u0]), torch.tensor([1 - u0])
    t = torch.tensor([-1.0, 1.0])
    d = torch.where(t >= 0, t * hi, t * lo)
    np.testing.assert_allclose((u0 + d).numpy(), [0.0, 1.0], atol=1e-12)
