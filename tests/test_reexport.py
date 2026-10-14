"""segment/convert/screen are thin re-exports of twobunch-s2e-rl's xtcav_segment, on
released cheetah-accelerator 0.8.4, and reproduce the Stage-A golden numbers."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from xtcav_gpsr_tuning import convert, paths, screen, segment  # noqa: E402

cheetah = pytest.importorskip("cheetah")


def test_modules_are_reexports():
    import xtcav_segment.convert
    import xtcav_segment.screen
    import xtcav_segment.segment

    assert segment.build_segment_from_spec is xtcav_segment.segment.build_segment_from_spec
    assert convert.cloud_to_beam is xtcav_segment.convert.cloud_to_beam
    assert screen.render_two_bunch is xtcav_segment.screen.render_two_bunch


def test_released_cheetah_not_the_clone():
    assert cheetah.__version__ == "0.8.4"
    assert not Path(cheetah.__file__).resolve().is_relative_to(paths.CHEETAH_CLONE.resolve())


def test_segment_specs_identical_to_shared_copy():
    for p in sorted(paths.CONFIGS_DIR.glob("segment_*.json")):
        shared = paths.SEGMENT_SPECS_DIR / p.name
        assert shared.exists(), f"{p.name} missing from {paths.SEGMENT_SPECS_DIR}"
        assert p.read_bytes() == shared.read_bytes(), p.name


def test_stage_a_golden_baseline():
    """Stage-A Test 1 (exit->exit, V=0) and Test 4 streak S on the baseline Bmad reference
    beams reproduce results/stageA/validation_baseline.json (clone, 2026-07-30) exactly."""
    ref_dir = paths.DATA_DIR / "bmad_reference_baseline"
    golden = paths.RESULTS_DIR / "stageA" / "validation_baseline.json"
    if not (ref_dir / "beam_XTCAVF_off.h5").exists() or not golden.exists():
        pytest.skip("Stage-A reference beams / results not available")
    from pmd_beamphysics import ParticleGroup

    g = json.loads(golden.read_text())["tests"]
    spec = segment.load_spec(paths.CONFIGS_DIR / "segment_baseline.json")
    E_ref, L_tdc = spec["meta"]["ref_energy_eV"], spec["meta"]["tdc_length_m"]
    V = json.loads((ref_dir / "manifest.json").read_text())["voltage_V"]
    Px = ParticleGroup(str(ref_dir / "beam_XTCAVF_off.h5"))
    beam, _ = convert.particlegroup_to_beam(Px, E_ref, None)
    seg_exit, _ = segment.build_segment_from_spec(spec, start_after="XTCAVF")
    ref, _ = convert.particlegroup_to_beam(ParticleGroup(str(ref_dir / "beam_DTOTR_off.h5")), E_ref, None)
    cmp = convert.compare_beams(seg_exit.track(beam), ref, center=("tau",))
    for c in ("x", "px", "y", "py", "tau", "delta"):
        assert cmp[c]["rms"] == pytest.approx(g["1_pure_optics_exit_to_exit"][c]["rms"], rel=1e-12), c

    seg_full, _ = segment.build_segment_from_spec(spec)
    segment.set_tdc(seg_full, V, 0.0)
    out = seg_full.track(segment.back_drift(beam, L_tdc))
    z1 = -convert.C_LIGHT * (np.asarray(Px.t) - np.average(Px.t, weights=Px.weight))
    S = float(np.polyfit(z1, out.x.detach().numpy(), 1)[0])
    assert S == pytest.approx(g["4_streak_S"]["cheetah_fit"], rel=1e-12)
