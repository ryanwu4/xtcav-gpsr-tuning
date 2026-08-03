"""Stage-C C4b: flow-vs-Bmad image-gap floor at δ=0 across the 24 valtracks.

For each valtrack sample: render the Bmad DTOTR beams (plus & minus, 100k) and
the flow's prediction at the SAME campaign knobs through the identical
screen path; report the TV-distance gap with a noise decomposition:
  - flow_floor: two independent flow z-draws against each other
  - bmad_floor: two independent 10k subsamples of the 100k Bmad beam
  - gap: flow render vs 100k Bmad target
This is the irreducible Stage-D residual from surrogate error (plan C4).

  PYTHONPATH=$PWD/src:$TWOBUNCH/src CUDA_VISIBLE_DEVICES=0 \
    python scripts/run_stageC_imagegap.py

Outputs -> results/stageC/imagegap.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from xtcav_gpsr_tuning.fit import ForwardModel, load_setup  # noqa: E402
from xtcav_gpsr_tuning.targets import load_dtotr_target, target_charge_split  # noqa: E402

VALTRACKS = paths.TWOBUNCH_ROOT / "data" / "xtcav23_full_valtracks"
CAMPAIGN = paths.TWOBUNCH_ROOT / "data" / "xtcav23_full"
OUT = REPO / "results" / "stageC"


def tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().sum())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_setup(device=device)
    manifest = json.loads((VALTRACKS / "valtracks_manifest.json").read_text())
    norm = setup.norm
    keys, lo, hi = norm["knob_keys"], np.array(norm["knob_low"]), np.array(norm["knob_high"])
    n_p = int(cfg["n_particles"])

    rows = []
    for tag, rec in sorted(manifest.items()):
        knobs_phys = rec["knobs"]
        u = (np.array([knobs_phys[k] for k in keys]) - lo) / (hi - lo)
        base = torch.tensor(u, dtype=torch.float64).unsqueeze(0).to(device)

        model = ForwardModel(setup, base, rec["voltage_V"], n_particles=n_p,
                             seed=0).to(device)
        model2 = ForwardModel(setup, base, rec["voltage_V"], n_particles=n_p,
                              seed=1).to(device)
        with torch.no_grad():
            f1, s1 = model()   # (2,1,ny,nx) [plus, minus]
            f2, _ = model2()

        row = {"tag": tag}
        for i, case in enumerate(("plus", "minus")):
            h5 = VALTRACKS / f"{tag}_DTOTR_{case}.h5"
            t100 = load_dtotr_target(h5, setup.grid).to(device)
            ta = load_dtotr_target(h5, setup.grid, subsample=10000, seed=11).to(device)
            tb = load_dtotr_target(h5, setup.grid, subsample=10000, seed=12).to(device)
            row[f"{case}_gap_tv"] = tv(f1[i, 0], t100)
            row[f"{case}_flow_floor_tv"] = tv(f1[i, 0], f2[i, 0])
            row[f"{case}_bmad10k_floor_tv"] = tv(ta, tb)
        row["split_flow"] = float(s1[0, 0])
        row["split_bmad"] = target_charge_split(VALTRACKS / f"{tag}_DTOTR_plus.h5")
        rows.append(row)
        print(f"{tag}: gap+ {row['plus_gap_tv']:.3f} (flow floor "
              f"{row['plus_flow_floor_tv']:.3f}, bmad10k floor "
              f"{row['plus_bmad10k_floor_tv']:.3f}) split {row['split_flow']:.3f} "
              f"vs {row['split_bmad']:.3f}", flush=True)

    gaps = [r["plus_gap_tv"] for r in rows] + [r["minus_gap_tv"] for r in rows]
    floors = [r["plus_flow_floor_tv"] for r in rows]
    out = {"rows": rows,
           "summary": {
               "gap_tv_median": float(np.median(gaps)),
               "gap_tv_p90": float(np.quantile(gaps, 0.9)),
               "flow_floor_tv_median": float(np.median(floors)),
               "split_err_median": float(np.median([abs(r["split_flow"] - r["split_bmad"])
                                                    for r in rows])),
               "n_particles": n_p,
               "bandwidth_um": cfg["bandwidth_um"],
           }}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "imagegap.json").write_text(json.dumps(out, indent=1))
    print("summary:", json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
