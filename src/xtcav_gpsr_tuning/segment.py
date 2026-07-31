"""XTCAVF->DTOTR differentiable segment: spec export (PyTao side) + Cheetah build.

Two halves, split across conda envs:

* ``extract_segment_spec(tao, ...)`` runs under ``bmad-qpad-dev`` (pytao) on a
  *configured* Tao (after setLattice), and serializes the resolved element
  sequence to a JSON spec via the bridge's ``extract_elements``.
* ``build_segment_from_spec(spec, ...)`` runs anywhere with torch + the Cheetah
  clone, and builds a differentiable ``cheetah.Segment``.

Design rules (from the approved plan):
* crab_cavity -> cheetah.TransverseDeflectingCavity with **PHI0 passed through
  unchanged** (both are in rad/2pi units — do NOT apply the bridge's lcavity
  degrees+sign-flip conversion; critic finding M1).
* No monkey-patching: we run our own build loop calling the bridge's
  ``_make_element`` per element, pre-intercepting crab_cavity (critic M6).
* ``tracking_method='drift_kick_drift'`` pinned on every element that supports
  it (critic M4: Cheetah's default linear maps carry no per-particle
  chromaticity, and this line is chromatically hair-triggered).
* Plane bookkeeping: the exported spec is START-INCLUSIVE from XTCAVF, i.e. it
  CONTAINS the TDC. Saved Bmad artifact beams are at the element EXIT, so use
  ``start_after='XTCAVF'`` when tracking exit-plane beams (critic C1 — never
  track an already-streaked beam through a TDC-containing segment).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import paths

TDC_NAME = "XTCAVF"


# --------------------------------------------------------------------------- #
# Export side (bmad-qpad-dev: pytao + the bridge)
# --------------------------------------------------------------------------- #
def extract_segment_spec(tao, start: str = TDC_NAME, end: str = "DTOTR", meta: dict | None = None) -> dict:
    """Extract a JSON-serializable segment spec from a configured Tao instance.

    The element list is start-inclusive (contains ``start`` itself). Meta
    records the lattice reference energy and reference time at both planes
    (exit-plane, matching Bmad's ele.s / saved-beam convention).
    """
    paths.ensure_import_roots()
    from pytao_to_cheetah import extract_elements

    elements = extract_elements(tao, start, end)

    spec_meta = dict(meta or {})
    for label, name in (("start", start), ("end", end)):
        head = tao.ele_head(name)
        spec_meta[f"{label}_name"] = name
        spec_meta[f"{label}_s_m"] = float(head["s"])
        # Lattice reference time at the element exit — needed to reference tau
        # for absolute RF phase (convert.py contract). ele_head exposes it as
        # 'ref_time' in current pytao; fall back to lat_list if absent.
        if "ref_time" in head:
            spec_meta[f"{label}_ref_time_s"] = float(head["ref_time"])
        else:  # pragma: no cover - pytao version fallback
            rt = tao.lat_list(name, "ele.ref_time", flags="-array_out -no_slaves")
            spec_meta[f"{label}_ref_time_s"] = float(rt[0])
    # Reference energy: constant through this segment (no cavities); use the
    # start element's exit-plane e_tot.
    spec_meta["ref_energy_eV"] = float(elements[0]["e_tot"])

    return {"start": start, "end": end, "meta": spec_meta, "elements": elements}


def save_spec(spec: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(spec, indent=1, sort_keys=True))


def load_spec(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Build side (torch + Cheetah clone)
# --------------------------------------------------------------------------- #
def _make_tdc(ed: dict, factory_kwargs: dict):
    """crab_cavity -> TransverseDeflectingCavity. PHI0 passthrough (rad/2pi)."""
    import cheetah

    a = ed["attrs"]

    def t(v):
        return torch.tensor(float(v), **factory_kwargs)

    return cheetah.TransverseDeflectingCavity(
        length=t(a.get("l", 0.0)),
        voltage=t(a.get("voltage", 0.0)),
        phase=t(a.get("phi0", 0.0)),  # rad/2pi in BOTH Bmad and Cheetah TDC — no conversion
        frequency=t(a.get("rf_frequency", 0.0)),
        tilt=t(a.get("tilt", 0.0)),
        name=ed["name"],
        sanitize_name=True,
    )


def build_segment_from_spec(
    spec: dict,
    start_after: str | None = None,
    device=None,
    dtype=torch.float64,
    name: str | None = None,
):
    """Build a differentiable Cheetah Segment from an exported spec.

    :param start_after: drop elements up to AND INCLUDING this name. Use
        ``start_after='XTCAVF'`` to get the exit-plane segment (for tracking
        saved exit-plane beams); ``None`` keeps the TDC (entrance-plane
        segment, for the bracket test / the fit-time forward model).
    :return: (segment, ConversionReport)
    """
    paths.ensure_import_roots()
    import cheetah
    from pytao_to_cheetah import ConversionReport, _make_element, _set_tracking_method

    factory_kwargs = {
        "device": device or torch.get_default_device(),
        "dtype": dtype or torch.get_default_dtype(),
    }

    eds = list(spec["elements"])
    if start_after is not None:
        idx = [i for i, ed in enumerate(eds) if ed["name"] == start_after]
        if not idx:
            raise ValueError(f"start_after element {start_after!r} not in spec")
        eds = eds[idx[0] + 1 :]

    report = ConversionReport(n_in=len(eds))
    if eds:
        report.bmad_length = eds[-1]["s"] - (eds[0]["s"] - float(eds[0]["attrs"].get("l", 0.0)))

    elements = []
    for ed in eds:
        if ed["key"].lower() == "crab_cavity":
            elements.append(_make_tdc(ed, factory_kwargs))
        else:
            elements.append(_make_element(ed, report, factory_kwargs))

    seg_name = name or f"{spec['start']}_to_{spec['end']}"
    segment = cheetah.Segment(elements=elements, name=seg_name, sanitize_name=True)
    _set_tracking_method(segment, "drift_kick_drift")

    flat = list(segment.flattened().elements)
    counts: dict = {}
    total_len = 0.0
    for e in flat:
        counts[type(e).__name__] = counts.get(type(e).__name__, 0) + 1
        if getattr(e, "length", None) is not None:
            total_len += float(e.length)
    report.type_counts = counts
    report.cheetah_length = total_len
    return segment, report


def element_by_name(segment, name: str):
    """Fetch an element from a (possibly nested) Segment by its Bmad name."""
    for e in segment.flattened().elements:
        if getattr(e, "name", None) == name:
            return e
    raise KeyError(f"element {name!r} not found in segment")


def set_tdc(segment, voltage_V: float, phase_deg: float = 0.0) -> None:
    """Set the TDC like the sim scripts set Bmad (VOLTAGE in V, PHI0 = deg/360)."""
    tdc = element_by_name(segment, TDC_NAME)
    tdc.voltage = torch.as_tensor(
        float(voltage_V), device=tdc.voltage.device, dtype=tdc.voltage.dtype
    )
    tdc.phase = torch.as_tensor(
        float(phase_deg) / 360.0, device=tdc.phase.device, dtype=tdc.phase.dtype
    )


def back_drift(beam, length_m: float):
    """Exact inverse drift (Bmad-X map with negative length) — used to synthesize
    the TDC ENTRANCE beam from a saved exit-plane V=0 beam (critic C1 bracket)."""
    paths.ensure_import_roots()
    import cheetah

    d = cheetah.Drift(
        length=torch.tensor(-float(length_m), dtype=beam.particles.dtype, device=beam.particles.device)
    )
    if "drift_kick_drift" in type(d).supported_tracking_methods:
        d.tracking_method = "drift_kick_drift"
    return d.track(beam)


def numerical_rmatrix(segment, ref_energy_eV: float, species_name: str = "electron", eps: float = 1e-6):
    """First-order transfer matrix of the segment by central finite differences.

    DKD elements expose no closed-form map (critic M4), so we linearize
    numerically about the reference orbit. Coordinates are Cheetah's
    (x, px, y, py, tau, delta); for R12/R16/R36 comparisons against Bmad's
    (x, px, y, py, z, pz) this is direct — z/pz enter only rows/columns 5,6,
    where z = -beta*tau flips signs (beta ~ 1 - 1.3e-9 at 10 GeV).
    """
    paths.ensure_import_roots()
    import cheetah
    from cheetah.particles import Species

    dtype = torch.float64
    species = Species(species_name, dtype=dtype)
    n = 6
    probes = torch.zeros((2 * n + 1, 7), dtype=dtype)
    probes[:, 6] = 1.0
    for j in range(n):
        probes[2 * j, j] = +eps
        probes[2 * j + 1, j] = -eps

    beam = cheetah.ParticleBeam(
        particles=probes,
        energy=torch.tensor(float(ref_energy_eV), dtype=dtype),
        particle_charges=torch.ones(2 * n + 1, dtype=dtype),
        survival_probabilities=torch.ones(2 * n + 1, dtype=dtype),
        species=species,
    )
    out = segment.track(beam).particles[..., :6]
    R = torch.zeros((n, n), dtype=dtype)
    for j in range(n):
        R[:, j] = (out[2 * j] - out[2 * j + 1]) / (2 * eps)
    return R.numpy()
