"""Sibling-repo resolution for the xtcav-gpsr-tuning workspace.

This repo is used via PYTHONPATH (never pip-installed) and imports siblings
read-only:
  - twobunch-s2e-rl via its src/: the XTCAV forward model (``xtcav_segment``:
    segment / convert / screen, re-exported by this package's modules of the
    same names), the surrogate model, and xtcav_metrics
  - FACET2-S2E (importable as FACET2_S2E only in the bmad-qpad-dev env)

Cheetah is the released ``cheetah-accelerator==0.8.4`` from the environment
(installed in slac-rl). The 0.8.3-dev source clone and the PyTao->Cheetah
bridge are no longer put on sys.path (the bridge functions the segment needs are
vendored in ``xtcav_segment._bridge``); their paths stay below for reference.
For the spec export under bmad-qpad-dev (no cheetah there), put a 0.8.4 copy on
PYTHONPATH, e.g. ``twobunch-s2e-rl/.vendor/cheetah084``.

`ensure_import_roots()` must be called before importing ``xtcav_segment`` or
``twobunch_s2e_rl``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../xtcav-gpsr-tuning
WORKSPACE = REPO_ROOT.parent  # .../photoinjector-rl

CHEETAH_CLONE = WORKSPACE / "prev_work" / "cheetah_analysis" / "cheetah"  # reference only
BRIDGE_DIR = WORKSPACE / "prev_work" / "cheetah_analysis" / "bmad_cheetah_bridge"  # reference only
FACET2_S2E_ROOT = WORKSPACE / "FACET2-S2E"
TWOBUNCH_ROOT = WORKSPACE / "twobunch-s2e-rl"
TWOBUNCH_SRC = TWOBUNCH_ROOT / "src"
# The one shared copy of the exported segment specs. configs/segment_*.json here are
# kept byte-identical (tests/test_reexport.py) so existing yaml/script paths still work.
SEGMENT_SPECS_DIR = TWOBUNCH_ROOT / "configs" / "segment"
GPSR_DIR = WORKSPACE / "gpsr"  # pattern reference only, never imported

DATA_DIR = REPO_ROOT / "data"
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"


def ensure_import_roots() -> None:
    """Prepend twobunch-s2e-rl src to sys.path (for xtcav_segment / twobunch_s2e_rl)."""
    for p in (TWOBUNCH_SRC,):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
