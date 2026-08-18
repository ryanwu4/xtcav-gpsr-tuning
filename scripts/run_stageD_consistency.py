"""Stage-D D3.7: is δ̂ a property of the MACHINE or of the scan region?

Refits the winning orientation on independent subsets of the 33-step scan and
compares δ̂. Two splits with different logic:

  low / deep   — first 17 vs last 16 steps: tests whether δ̂ is stable across
                 compression state (a machine error must be; a model-deficiency
                 absorber need not be)
  even / odd   — interleaved steps: same statistics and the same compression
                 range, so it isolates pure statistical stability as the control
                 for the low/deep comparison

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=$PWD/src:.../twobunch-s2e-rl/src python scripts/run_stageD_consistency.py

Output -> results/stageD/15673/fit/consistency.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from xtcav_gpsr_tuning import paths  # noqa: E402

sys.path.insert(0, str(paths.TWOBUNCH_SRC))
from run_stageD_fit import CONV_TV_MAX, fit_once, orient_targets  # noqa: E402
from xtcav_gpsr_tuning.stage_d import (check_provenance, conditioning,  # noqa: E402
                                       load_stage_d, load_targets)

OUT = REPO / "results" / "stageD" / "15673" / "fit"
PRECAL = REPO / "results" / "stageD" / "15673" / "precal"


def subset(tgt, rows):
    """Slice every step-indexed target quantity onto `rows`."""
    return {
        "images": tgt["images"][:, rows],
        "splits": tgt["splits"][:, rows],
        "weights": tgt["weights"][:, rows],
        "off": tgt["off"][rows],
        "off_weights": (None if tgt.get("off_weights") is None
                        else tgt["off_weights"][rows]),
        "mask": tgt["mask"],
        "floors": tgt["floors"][:, rows],
    }


def main():
    lam_json = json.loads((PRECAL / "lambda.json").read_text())
    check_provenance(lam_json, what="precal/lambda.json")
    lam = float(lam_json["lambda_star"])
    sel = json.loads((OUT / "orient.json").read_text())
    key = sel["winner"]
    fr, sw = sel["runs"][key]["flip_rows"], sel["runs"][key]["swap_signs"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, setup = load_stage_d(device=device)
    base = conditioning(setup, cfg)
    real = load_targets(cfg, device=device)
    tgt = orient_targets(real, fr, sw)
    n_steps = base.shape[0]
    scan = real["scan_vals_deg"]
    print(f"orientation {key}, lambda* {lam:g}, {n_steps} steps", flush=True)

    splits = {
        "low":  list(range(0, 17)),
        "deep": list(range(17, n_steps)),
        "even": list(range(0, n_steps, 2)),
        "odd":  list(range(1, n_steps, 2)),
    }
    results = {"orientation": key, "lambda": lam, "scan_vals_deg": scan.tolist(),
               "splits": {}}
    for name, rows in splits.items():
        r = fit_once(setup, cfg, base[rows], subset(tgt, rows), lam, seed=0)
        results["splits"][name] = {
            "rows": rows,
            "scan_range_deg": [float(scan[rows[0]]), float(scan[rows[-1]])],
            "img_loss": r["final_img_loss"], "off_loss": r["final_off_loss"],
            "pulls": r["pulls"], "delta_phys": r["delta_phys"],
            "sdelta": r["sdelta"], "ex": r["ex"], "nu_y": r["nu_y"],
            "l2_shift_deg": r["l2_shift_deg"], "dtr": r["dtr"],
            "converged": bool(r["final_img_loss"] < CONV_TV_MAX),
        }
        print(f"  {name:5s} steps {rows[0]}-{rows[-1]} (n={len(rows)}): "
              f"img {r['final_img_loss']:.4f} sdelta {r['sdelta']:.3f} "
              f"ex {r['ex']:.3f} max|pull| "
              f"{max(abs(v) for v in r['pulls'].values()):.2f}", flush=True)

    knobs = list(results["splits"]["low"]["pulls"])
    for a, b in (("low", "deep"), ("even", "odd")):
        pa = np.array([results["splits"][a]["pulls"][k] for k in knobs])
        pb = np.array([results["splits"][b]["pulls"][k] for k in knobs])
        results[f"delta_{a}_vs_{b}"] = {
            "per_knob": dict(zip(knobs, (pa - pb).tolist())),
            "rms_pull_difference": float(np.sqrt(np.mean((pa - pb) ** 2))),
            "max_abs_difference": float(np.abs(pa - pb).max()),
            "corr": float(np.corrcoef(pa, pb)[0, 1]),
        }
        m = results[f"delta_{a}_vs_{b}"]
        print(f"  {a} vs {b}: rms Δpull {m['rms_pull_difference']:.3f}"
              f"  max {m['max_abs_difference']:.3f}  corr {m['corr']:+.3f}",
              flush=True)

    (OUT / "consistency.json").write_text(json.dumps(results, indent=1))
    print(f"done -> {OUT / 'consistency.json'}")


if __name__ == "__main__":
    main()
