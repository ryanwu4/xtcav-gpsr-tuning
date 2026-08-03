"""Evaluate Stage-C gates (ii)/(vi)/(vii) from the Bmad-truth recovery fits.

Gate (ii): on the PRE-REGISTERED identifiable subspace, |projection(δ̂) −
projection(δ_true)| ≤ ~1 (prior-scaled), with replica error bars, referenced to
the statistics-matched C3 self-recovery baseline (rms 0.078).
Gate (vi): kicker-in-truth scenario — fitted knob-δ pulls stay small (the
injected kicker has no DOF; ν_y must absorb its η_y signature).
Gate (vii): veff scenario — fitted V_eff recovers 24.2 MV within replica spread.

  PYTHONPATH=$PWD/src:$TWOBUNCH/src python scripts/finalize_stageC.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from xtcav_gpsr_tuning import paths  # noqa: E402

OUT = REPO / "results" / "stageC"
ident = json.loads((OUT / "selfrecovery" / "identifiability.json").read_text())
rec = json.loads((OUT / "bmadrecovery.json").read_text())
fitcfg_sigma = None

names = ident["param_names"]
V = np.array(ident["kept_directions"])
knob_names = [n.split(":", 1)[1] for n in names if n.startswith("delta:")]
nk = len(knob_names)

import yaml  # noqa: E402

cfg = yaml.safe_load((REPO / "configs" / "fit_stageC.yaml").read_text())
sigma_phys = {k: float(v) for k, v in cfg["sigma_phys"].items()}


def theta(pulls, nu_y=1.0, dtr=(0, 0, 0), veff_mv=0.0):
    th = np.zeros(len(names))
    for i, k in enumerate(knob_names):
        th[i] = pulls.get(k, 0.0)
    th[nk:nk + 3] = np.array(dtr) / 0.01
    th[nk + 3] = np.log(nu_y)
    th[nk + 4] = veff_mv
    return th


gates = {}
print("=== Gate (ii): Bmad-truth bias on the identifiable subspace ===")
worst = 0.0
for scen, data in rec["scenarios"].items():
    truth_pulls = {k: dv / sigma_phys[k] for k, dv in data["delta_true_phys"].items()
                   if k in sigma_phys}
    t_proj = V @ theta(truth_pulls)
    f = np.array([V @ theta(r["pulls"], r["nu_y"], r["dtr"],
                            (r["veff"] - 22e6) / 1e6) for r in data["fits"]])
    bias = f.mean(0) - t_proj
    spread = f.std(0)
    with np.printoptions(precision=2, suppress=True):
        print(f"  {scen:7s}: truth {t_proj} -> fit {f.mean(0)} "
              f"| bias {bias} (±{spread})")
    if scen in ("mover", "energy"):  # the knob-recovery scenarios
        worst = max(worst, float(np.abs(bias).max()))
gates["ii_worst_identifiable_bias"] = worst
gates["ii_pass"] = worst <= 1.0
print(f"  -> worst |bias| on knob scenarios: {worst:.2f} (prior-scaled; <=1 passes; "
      f"C3 baseline rms 0.078)")

print("\n=== Gate (vi): kicker elimination ===")
k = rec["scenarios"]["kicker"]
pulls = np.array([[abs(v) for v in r["pulls"].values()] for r in k["fits"]])
nus = [r["nu_y"] for r in k["fits"]]
print(f"  injected: {k['delta_true_phys']} (NO kicker DOF in fit)")
print(f"  max |knob pull| per replica: {pulls.max(1).round(2).tolist()}; "
      f"nu_y: {[round(n, 4) for n in nus]}")
gates["vi_max_pull"] = float(pulls.max())
gates["vi_pass"] = bool(pulls.max() <= 1.5)  # pre-registered ~2-60 um secondary scale
print(f"  -> pass (pulls <=1.5): {gates['vi_pass']}")

print("\n=== Gate (vii): V_eff recovery ===")
v = rec["scenarios"]["veff"]
veffs = [r["veff"] / 1e6 for r in v["fits"]]
truth_v = v["voltage_true_V"] / 1e6
print(f"  truth {truth_v:.2f} MV; fitted: {[round(x, 3) for x in veffs]} "
      f"(mean {np.mean(veffs):.3f} ± {np.std(veffs):.3f})")
gates["vii_veff_fit_MV"] = float(np.mean(veffs))
gates["vii_veff_truth_MV"] = truth_v
gates["vii_pass"] = bool(abs(np.mean(veffs) - truth_v) <= max(3 * np.std(veffs), 0.25))
print(f"  -> pass: {gates['vii_pass']}")

(OUT / "gates_final.json").write_text(json.dumps(gates, indent=1))
print("\nwrote", OUT / "gates_final.json")
