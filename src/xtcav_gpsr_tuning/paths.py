"""Sibling-repo resolution for the xtcav-gpsr-tuning workspace.

This repo is used via PYTHONPATH (never pip-installed) and imports four siblings
read-only:
  - the Cheetah 0.8.3-dev source clone (prev_work/cheetah_analysis/cheetah)
  - the PyTao->Cheetah bridge (prev_work/cheetah_analysis/bmad_cheetah_bridge)
  - twobunch-s2e-rl (surrogate model, xtcav_metrics) via its src/
  - FACET2-S2E (importable as FACET2_S2E only in the bmad-qpad-dev env)

`ensure_import_roots()` must be called before importing `cheetah` or
`pytao_to_cheetah`; the clone is inserted at the FRONT of sys.path so it wins
over any pip-installed cheetah.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../xtcav-gpsr-tuning
WORKSPACE = REPO_ROOT.parent  # .../photoinjector-rl

CHEETAH_CLONE = WORKSPACE / "prev_work" / "cheetah_analysis" / "cheetah"
BRIDGE_DIR = WORKSPACE / "prev_work" / "cheetah_analysis" / "bmad_cheetah_bridge"
FACET2_S2E_ROOT = WORKSPACE / "FACET2-S2E"
TWOBUNCH_ROOT = WORKSPACE / "twobunch-s2e-rl"
TWOBUNCH_SRC = TWOBUNCH_ROOT / "src"
GPSR_DIR = WORKSPACE / "gpsr"  # pattern reference only, never imported

DATA_DIR = REPO_ROOT / "data"
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"


def ensure_import_roots() -> None:
    """Prepend the Cheetah clone, the bridge, and twobunch-s2e-rl src to sys.path."""
    for p in (CHEETAH_CLONE, BRIDGE_DIR, TWOBUNCH_SRC):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
