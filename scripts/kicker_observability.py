"""Kicker observability study: do the 4 FF kickers have measurable objectives?

Part 1 — Bmad campaign data: absolute centroid swings at XTCAVF per kicker.
Part 2 — trained flow: learned centroid + incoming-dispersion responses per kicker.
Part 3 — through the Stage-A segment: eta_y and y-centroid at DTOTR per kicker,
          vs the d_nom uncertainty band and the mover lever arms.
"""

import json
import sys
from glob import glob

import numpy as np
import torch

sys.path.insert(0, "/home/rwu4/photoinjector-rl/twobunch-s2e-rl/src")
sys.path.insert(0, "/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning/src")

from twobunch_s2e_rl.datagen.sweep_params import XTCAV23_PARAMS  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402
from xtcav_gpsr_tuning import paths  # noqa: E402
from xtcav_gpsr_tuning.convert import cloud_to_beam  # noqa: E402
from xtcav_gpsr_tuning.segment import build_segment_from_spec, load_spec, set_tdc  # noqa: E402

paths.ensure_import_roots()

TB = "/home/rwu4/photoinjector-rl/twobunch-s2e-rl"
KEYS = list(XTCAV23_PARAMS)
KICKERS = ["XC1FFkG", "XC3FFkG", "YC1FFkG", "YC2FFkG"]
MOVER_REF = ["S1EL_yOffset", "S2ER_xOffset"]

# ---------------- Part 1: Bmad campaign absolute centroid swings ------------- #
print("=" * 72)
print("Part 1 — Bmad campaign: centroid swing across each kicker box (linear fit)")
rows = []
for camp in ("xtcav23_full", "xtcav23_pilot"):
    for fp in sorted(glob(f"{TB}/data/{camp}/sample_*.json")):
        d = json.load(open(fp))
        if d.get("is_baseline_repeat"):
            continue
        x = d["specs"]["XTCAVF"]
        if x.get("PDrive_median_x") is None:
            continue
        rows.append((
            [d["knobs"][k] for k in KEYS],
            x["PDrive_median_x"], x["PDrive_median_y"],
            x.get("PWitness_median_x"), x.get("PWitness_median_y"),
        ))
K = np.array([r[0] for r in rows])
print(f"  ({len(rows)} samples)")
for name in KICKERS + MOVER_REF:
    j = KEYS.index(name)
    lo, hi, _ = XTCAV23_PARAMS[name]
    span = hi - lo
    out = [name]
    for oi, lab in [(1, "drive_x"), (2, "drive_y")]:
        y = np.array([r[oi] for r in rows], float)
        m = np.isfinite(y)
        slope = np.polyfit(K[m, j], y[m], 1)[0]
        out.append(f"{lab} {abs(slope) * span * 1e6:7.1f} um")
    print("  {:14s} full-box swing: {}".format(out[0], ", ".join(out[1:])))

# ---------------- Part 2: flow-learned centroid + dispersion responses ------- #
print("\n" + "=" * 72)
print("Part 2 — trained flow: learned responses over each kicker dim "
      "(n=4096/point, others at box center)")
CKPT = f"{TB}/trained/xtcav23/checkpoints/best-epoch=498-val_loss=0.1747.ckpt"
model = TwoBunchFlow.load_from_checkpoint(CKPT, map_location="cpu").eval()
norm = json.load(open(f"{TB}/processed/xtcav23_combined_norm.json"))
E_MC2 = norm["electron_mc2_ev"]

ngrid, nrep, N = 13, 8, 4096
grid = np.linspace(0.02, 0.98, ngrid)

def cloud_stats(cloud):
    """charge-equal cloud (B,N,6) -> centroid_x/y [m], eta_y = cov(y,delta)/var(delta)."""
    x, y, pz = cloud[..., 0], cloud[..., 1], cloud[..., 5]
    delta = (pz - pz.mean(dim=-1, keepdim=True)) / pz.mean(dim=-1, keepdim=True)
    yc = y - y.mean(dim=-1, keepdim=True)
    eta = (yc * delta).mean(dim=-1) / delta.var(dim=-1)
    return x.mean(dim=-1), y.mean(dim=-1), eta

torch.manual_seed(0)
# noise floor: repeated identical center-point conditions
with torch.no_grad():
    Kc = torch.full((nrep, 23), 0.5)
    d = model.sample_bunch(Kc, 0, N).double()
    cx0, cy0, eta0 = cloud_stats(d)
nf = {"cx": cx0.std().item(), "cy": cy0.std().item(), "eta": eta0.std().item()}
print(f"  sampling noise floor (n={N}): centroid_x {nf['cx']*1e6:.1f} um, "
      f"centroid_y {nf['cy']*1e6:.1f} um, eta_y {nf['eta']*1e3:.1f} mm")

flow_swings = {}
for name in KICKERS + MOVER_REF:
    j = KEYS.index(name)
    Kk = torch.full((ngrid, 23), 0.5)
    Kk[:, j] = torch.tensor(grid, dtype=torch.float32)
    with torch.no_grad():
        dcl = model.sample_bunch(Kk, 0, N).double()
    cx, cy, eta = cloud_stats(dcl)
    sw = {
        "cx": np.polyfit(grid, cx.numpy(), 1)[0],
        "cy": np.polyfit(grid, cy.numpy(), 1)[0],
        "eta": np.polyfit(grid, eta.numpy(), 1)[0],
    }
    flow_swings[name] = sw
    print(f"  {name:14s} drive swings: cx {abs(sw['cx'])*1e6:7.1f} um "
          f"({abs(sw['cx'])/max(nf['cx'],1e-12):5.1f}x noise), "
          f"cy {abs(sw['cy'])*1e6:7.1f} um ({abs(sw['cy'])/max(nf['cy'],1e-12):5.1f}x), "
          f"eta_y {abs(sw['eta'])*1e3:6.1f} mm ({abs(sw['eta'])/max(nf['eta'],1e-12):5.1f}x)")

# ---------------- Part 3: through the segment to DTOTR ----------------------- #
print("\n" + "=" * 72)
print("Part 3 — DTOTR (through the Stage-A segment, V=0): eta_y + y-centroid "
      "swing per kicker (flow drive cloud, box extremes vs center)")
spec = load_spec("/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning/configs/segment_baseline.json")
E_ref = spec["meta"]["ref_energy_eV"]
seg, _ = build_segment_from_spec(spec, dtype=torch.float64)
set_tdc(seg, 0.0, 0.0)

def dtotr_stats(knobs_row):
    with torch.no_grad():
        torch.manual_seed(1)
        cl = model.sample_bunch(knobs_row.unsqueeze(0), 0, N).double()
        beam = cloud_to_beam(cl[0], E_ref, torch.full((N,), 1.2e-9 / N, dtype=torch.float64))
        out = seg.track(beam)
        y = out.y
        delta = out.particles[..., 5]
        eta = ((y - y.mean()) * (delta - delta.mean())).mean() / delta.var()
        return float(y.mean()), float(eta)

base_row = torch.full((23,), 0.5)
y0, eta_dt0 = dtotr_stats(base_row)
print(f"  center point: <y>_DTOTR {y0*1e3:+.3f} mm, eta_y(DTOTR) {eta_dt0*1e3:.1f} mm "
      f"(deployed d_nom = 59.31 mm; model-golden ~55.9; matched-config ~46-56)")
for name in KICKERS + MOVER_REF:
    j = KEYS.index(name)
    r_lo, r_hi = base_row.clone(), base_row.clone()
    r_lo[j], r_hi[j] = 0.02, 0.98
    ylo, elo = dtotr_stats(r_lo)
    yhi, ehi = dtotr_stats(r_hi)
    print(f"  {name:14s} full-box: d<y> {abs(yhi-ylo)*1e3:7.3f} mm, "
          f"d(eta_y) {abs(ehi-elo)*1e3:6.2f} mm "
          f"({abs(ehi-elo)/0.05931*100:5.1f}% of d_nom)")
EOF_MARKER_NOT_USED = None
