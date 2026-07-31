"""Converter contract tests (plan gotcha #6): t is authoritative, signs pinned."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from xtcav_gpsr_tuning import paths  # noqa: E402
from xtcav_gpsr_tuning.convert import C_LIGHT, particlegroup_to_beam  # noqa: E402

paths.ensure_import_roots()
pmd = pytest.importorskip("pmd_beamphysics")
from pmd_beamphysics import ParticleGroup  # noqa: E402

E_REF = 10e9  # eV


def make_pg(t, pz_eV=None, x=None):
    n = len(t)
    data = {
        "x": np.asarray(x if x is not None else np.zeros(n), dtype=float),
        "y": np.zeros(n),
        "z": np.zeros(n),
        "px": np.zeros(n),
        "py": np.zeros(n),
        "pz": np.asarray(pz_eV if pz_eV is not None else np.full(n, E_REF), dtype=float),
        "t": np.asarray(t, dtype=float),
        "status": np.ones(n, dtype=int),
        "weight": np.full(n, 1e-9 / n),
        "species": "electron",
    }
    return ParticleGroup(data=data)


def test_tau_sign_and_reference():
    """Earlier arrival (smaller t) = head = Bmad z>0 = Cheetah tau<0."""
    t0 = 3.3e-6
    dt = 1e-13  # 30 um
    P = make_pg([t0 - dt, t0, t0 + dt])
    beam, info = particlegroup_to_beam(P, E_REF, ref_time_s=t0)
    tau = beam.particles[..., 4].numpy()
    assert info["ref_time_s"] == t0
    # float64 ulp of t~3.3e-6 s is ~7e-22 s -> (t - t0) at dt=1e-13 is good to
    # ~7e-9 relative; that precision floor, not the formula, sets the tolerance
    np.testing.assert_allclose(tau, [-C_LIGHT * dt, 0.0, C_LIGHT * dt], rtol=3e-8)
    # and the beam's own stored z (as getBeamAtElement writes it) is -c*delta_t = -tau
    P.z = -C_LIGHT * (P.t - np.average(P.t, weights=P.weight))
    np.testing.assert_allclose(P.z, -tau, rtol=1e-9, atol=1e-18)


def test_nonzero_t_default_reference_is_weighted_mean():
    """With ref_time_s=None the charge-weighted mean arrival is used (z-vs-t trap)."""
    t = np.array([3.3e-6, 3.3e-6 + 2e-13])
    P = make_pg(t)
    P.weight = np.array([3e-9, 1e-9])
    beam, info = particlegroup_to_beam(P, E_REF)
    t_mean = np.average(t, weights=P.weight)
    assert abs(info["ref_time_s"] - t_mean) < 1e-19
    tau = beam.particles[..., 4].numpy()
    np.testing.assert_allclose(tau, C_LIGHT * (t - t_mean), rtol=1e-9, atol=1e-15)


def test_matches_cheetah_importer_when_t_is_relative():
    """Cheetah's own from_openpmd_particlegroup uses tau = +c*t (absolute).
    On a group whose t is already relative (mean 0) the two agree exactly,
    which pins our convention to Cheetah's."""
    import cheetah

    t = np.array([-1e-13, 0.0, 1e-13])
    P = make_pg(t, x=[1e-6, 0.0, -1e-6])
    ours, _ = particlegroup_to_beam(P, E_REF, ref_time_s=0.0)
    theirs = cheetah.ParticleBeam.from_openpmd_particlegroup(
        P, energy=torch.tensor(E_REF, dtype=torch.float64), dtype=torch.float64
    )
    np.testing.assert_allclose(
        ours.particles[..., :6].numpy(), theirs.particles[..., :6].numpy(),
        rtol=1e-12, atol=1e-18,
    )


def test_delta_uses_lattice_reference_energy():
    P = make_pg([0.0, 0.0], pz_eV=[E_REF, 1.001 * E_REF])
    beam, info = particlegroup_to_beam(P, E_REF, ref_time_s=0.0)
    delta = beam.particles[..., 5].numpy()
    p0c = info["p0c_eV"]
    E = np.asarray(P.energy)
    np.testing.assert_allclose(delta, (E - E_REF) / p0c, rtol=1e-12)
