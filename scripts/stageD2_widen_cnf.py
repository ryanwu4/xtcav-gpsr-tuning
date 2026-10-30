"""Stage-D2 wide-box fine-tune, step 1: widen the shipped c_b3_ema checkpoint to the union frame
of the C_B3 table + the wide-box campaign, and save it as a `twobunch-cnf-train --init-ckpt`
warm start. The shipped bundle is read, never modified (the wide model is fit-only).

The frame comes from the merged table's norm.json (what train.check_knob_frame compares
against) or, before that table exists, from a union sweep-set name of the stageD2-widebox
worktree. Verification: at random points inside the OLD box, the widened model's observables
equal the original's (same frozen base samples) up to float32 rounding.

  PYTHONPATH=$PWD/src python scripts/stageD2_widen_cnf.py \
      --norm <twobunch>/processed/<merged>_norm.json --out <twobunch>/trained/<run>/widened.ckpt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

import twobunch_cnf
from twobunch_cnf.artifacts import flow_knob_space
from twobunch_cnf.bundle import bundle_dir, read_bundle
from twobunch_cnf.knobs import KnobSpace
from twobunch_cnf.model import load_flow
from twobunch_cnf.widen import widen_conditioning

WT_SRC = Path(__file__).resolve().parents[2] / "twobunch-s2e-rl/.worktrees/stageD2_widebox/src"
KEYS = ("bunch_spacing", "drive_sigma_z", "witness_sigma_z", "drive_norm_emit_x",
        "energy_difference", "T_drive", "T_witness", "p_surv")


@torch.no_grad()
def verify(old, wide, n_points=16, n=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    so, sw = flow_knob_space(old), flow_knob_space(wide)
    phys = so.denormalize(torch.rand(n_points, so.n, generator=g, dtype=torch.float64))
    z = (torch.randn(n, old.latent_dim, generator=g), torch.randn(n, old.latent_dim, generator=g))
    # the beam latent is drawn inside observables(): the same generator seed gives both models the
    # same eps, so s = mu(knobs) + sigma(knobs) eps differs only through the remapped prior layer
    a = old.observables(so.normalize(phys).float(), n=n, z_drive=z[0], z_witness=z[1],
                        generator=torch.Generator().manual_seed(seed + 1))
    b = wide.observables(sw.normalize(phys).float(), n=n, z_drive=z[0], z_witness=z[1],
                         generator=torch.Generator().manual_seed(seed + 1))
    return {k: float(((b[k] - a[k]).abs() / a[k].abs().max().clamp_min(1e-30)).max()) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="c_b3_ema")
    ap.add_argument("--norm", help="merged table norm.json (the training frame)")
    ap.add_argument("--sweep-set", default="xtcav23e_l0b2_inj12+xtcav23w_l0b2_inj12w",
                    help="union sweep set, used when --norm is not given")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = read_bundle(a.bundle)
    ck = bundle_dir(a.bundle) / meta["checkpoint"]["file"]
    old = load_flow(ck)
    if a.norm:
        new = KnobSpace.from_norm_json(a.norm)
    else:
        sys.path.insert(0, str(WT_SRC))
        from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
        keys, lo, hi, _ = resolve_sweep_set(a.sweep_set)
        new = KnobSpace.from_lists(keys, lo, hi)
    so = flow_knob_space(old)
    changed = {k: ((so.low[i], so.high[i]), (new.low[new.index(k)], new.high[new.index(k)]))
               for i, k in enumerate(so.keys)
               if (so.low[i], so.high[i]) != (new.low[new.index(k)], new.high[new.index(k)])}
    wide = widen_conditioning(old, new)
    err = verify(old, wide)
    print("widened:", json.dumps(changed))
    print("max rel. observable difference in the old box:", json.dumps(err))
    if max(err.values()) > 1e-4:
        sys.exit("widened model does not reproduce the original")
    src = torch.load(ck, map_location="cpu", weights_only=False)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": wide.state_dict(), "hyper_parameters": dict(wide.hparams),
                "pytorch-lightning_version": src.get("pytorch-lightning_version"),
                "stageD2_widen": {"source": str(ck), "sha256": meta["checkpoint"]["sha256"],
                                  "twobunch_cnf": twobunch_cnf.__version__, "changed": changed,
                                  "verify": err}}, out)
    print(out)


if __name__ == "__main__":
    main()
