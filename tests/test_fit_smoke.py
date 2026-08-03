"""Fit-harness smoke: recover one injected mover δ (2σ_j) through a flow→flow
3-step mini-scan with an INDEPENDENT target z-draw (never the fit's own buffer).
Marked slow-ish (~2-4 min CPU); runs by default since it is the load-bearing check."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtcav_gpsr_tuning import paths  # noqa: E402

CKPT = paths.TWOBUNCH_ROOT / "trained/xtcav23/checkpoints/best-epoch=498-val_loss=0.1747.ckpt"
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="trained xtcav23 ckpt absent")


def test_recover_single_mover_delta():
    sys.path.insert(0, str(paths.TWOBUNCH_SRC))
    from xtcav_gpsr_tuning.fit import ForwardModel, load_setup, run_fit

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_setup(device=device)
    knob = "S2ER_xOffset"
    j = setup.fit_knobs.index(knob)
    sigma_n = float(setup.sigma_norm[j])

    base = torch.full((3, 23), 0.5, dtype=torch.float64)
    base[:, setup.norm["knob_keys"].index("L2PhaseSet")] = torch.tensor([0.45, 0.5, 0.55])

    # NOTE knob choice: S2ER_xOffset carries relative-offset + spacing signatures
    # that SURVIVE COM-centering. A y-mover (e.g. S1EL_yOffset) is near-degenerate
    # with the free nu_y on a short scan, so MAP correctly shrinks it toward 0 —
    # verified during Stage-C bring-up; that degeneracy belongs to identifiability.py,
    # not to this machinery smoke.
    # truth: 2σ_j on the one mover, rendered with an INDEPENDENT z-draw (seed 99)
    truth = ForwardModel(setup, base, cfg["voltage_V"], n_particles=4096, seed=99).to(device)
    with torch.no_grad():
        truth.delta_raw[j] = torch.atanh(torch.tensor(2.0 * sigma_n
                                                      / float(setup.bound_norm[j])))
        t_imgs, t_splits = truth()

    model = ForwardModel(setup, base, cfg["voltage_V"], n_particles=4096, seed=0).to(device)
    res = run_fit(model, t_imgs.detach(), t_splits.detach(), lam=1e-3,
                  iters=120, lr=0.05)

    span = setup.norm["knob_high"][setup.knob_index[j]] - \
        setup.norm["knob_low"][setup.knob_index[j]]
    truth_phys = 2.0 * sigma_n * span
    got = res["delta_phys"][knob]
    # recover the right sign and within ~1σ_j of the injected value
    assert got * truth_phys > 0, f"wrong sign: {got} vs {truth_phys}"
    assert abs(got - truth_phys) < 1.2 * sigma_n * span, (got, truth_phys)
    # other movers stay small (below ~1.5σ each)
    for k, v in res["pulls"].items():
        if k != knob and k.endswith("Offset"):
            assert abs(v) < 1.5, (k, v)
