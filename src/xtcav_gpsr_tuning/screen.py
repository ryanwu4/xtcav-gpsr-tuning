"""Re-export of ``xtcav_segment.screen`` (twobunch-s2e-rl/src/xtcav_segment).

Differentiable DTOTR2 KDE screen. Moved to twobunch-s2e-rl (architecture plan phase 2);
this module keeps every name the fit code and scripts import.
"""

from . import paths

paths.ensure_import_roots()  # puts twobunch-s2e-rl/src on sys.path

from xtcav_segment.screen import (  # noqa: E402,F401
    NX,
    NY,
    PIXEL_UM,
    ScreenGrid,
    _weighted_mean,
    charge_split,
    com_center_image,
    image_com,
    image_zlobe_split,
    render_image,
    render_two_bunch,
    shift_image,
    soft_major_lobe_anchor,
)
