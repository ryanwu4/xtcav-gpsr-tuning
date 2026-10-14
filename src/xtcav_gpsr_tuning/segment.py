"""Re-export of ``xtcav_segment.segment`` (twobunch-s2e-rl/src/xtcav_segment).

The XTCAVF->DTOTR forward model (spec export + Cheetah build) moved to twobunch-s2e-rl
(architecture plan phase 2) and runs on released cheetah-accelerator 0.8.4; this module
keeps every name the fit code and scripts import. See xtcav_segment/segment.py for the
design rules (DKD pinned, TDC PHI0 passthrough, start_after='XTCAVF' for exit-plane beams).
"""

from . import paths

paths.ensure_import_roots()  # puts twobunch-s2e-rl/src on sys.path

from xtcav_segment.segment import (  # noqa: E402,F401
    SEGMENT_QUADS,
    SPEC_DIR,
    TDC_NAME,
    _make_tdc,
    back_drift,
    build_segment_from_spec,
    element_by_name,
    extract_segment_spec,
    load_spec,
    numerical_rmatrix,
    quad_k1_from_kG,
    save_spec,
    set_quads_kG,
    set_tdc,
    spec_path,
)
