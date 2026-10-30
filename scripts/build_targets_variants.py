"""Stage-D2: E300_15673 target variants that differ ONLY in the off-shot
row-distortion correction applied to the real frames.

  deployed  the Stage D targets (results/stageD/15673/targets.npz): per-step
            references from the deployed classifier's ix_off, which the 08-07
            audit found 87% streaked (348/400 shots at 82-98% amplitude)
  clean     references from amplitude-gated true-off shots (amp < 1), pooled
            over ±2 steps exactly like the off targets (1-5 clean shots per step
            otherwise), global clean average as the fallback
  none      no row correction at all (for the 'neither side' arm)

Everything else is build_targets verbatim. CPU, slac-rl:

  PYTHONPATH=$PWD/src MPLBACKEND=Agg python scripts/build_targets_variants.py --variant clean
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning import targets_real as tr  # noqa: E402

OUT = REPO / "results" / "stageD2" / "15673"


def clean_refs(a, cls, steps_per_shot, pool=2, off_amp_max=1.0):
    xtcav, _ = tr._import_deployed()
    amp = np.asarray(cls["amp"], float)
    ix_off = np.asarray(cls["ix_off"], int)
    clean = ix_off[amp[ix_off] < off_amp_max]

    def ref(inds, step):
        frames = [np.asarray(a.get_image(int(i)).data, float) for i in inds]
        frames = [f for f in frames if f.sum() > 0]
        avg = np.mean(frames, axis=0)
        return {"off_avg": avg, "distortion_px": xtcav.fit_row_distortion(avg),
                "n_used": len(frames), "step_idx": step}

    refs = {None: ref(clean, None)}
    for step in np.unique(steps_per_shot[steps_per_shot >= 0]):
        sel = clean[np.abs(steps_per_shot[clean] - step) <= pool]
        if sel.size:
            refs[int(step)] = ref(sel, int(step))
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "none"], required=True)
    args = ap.parse_args()
    orig = tr.load_dataset

    def patched(name):
        ds, cls, sps, a, refs = orig(name)
        refs = {} if args.variant == "none" else clean_refs(a, cls, sps)
        return ds, cls, sps, a, refs

    tr.load_dataset = patched
    res = tr.build_targets("E300_15673")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"targets_rowref_{args.variant}.npz"
    np.savez_compressed(path, **res)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
