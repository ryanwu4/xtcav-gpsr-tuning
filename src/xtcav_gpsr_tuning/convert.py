"""Re-export of ``xtcav_segment.convert`` (twobunch-s2e-rl/src/xtcav_segment).

ParticleGroup / flow cloud <-> cheetah.ParticleBeam conversion (t authoritative,
tau = -z for flow clouds, beam-mean RF phase reference). Moved to twobunch-s2e-rl
(architecture plan phase 2); this module keeps every name the fit code and scripts import.
"""

from . import paths

paths.ensure_import_roots()  # puts twobunch-s2e-rl/src on sys.path

from xtcav_segment.convert import (  # noqa: E402,F401
    C_LIGHT,
    COORDS,
    _species,
    beam_to_cloud,
    cloud_to_beam,
    compare_beams,
    particlegroup_to_beam,
    weighted_stats,
)
