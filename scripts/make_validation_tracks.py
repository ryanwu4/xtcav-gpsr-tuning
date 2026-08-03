"""Generate held-out Bmad ground truth for a subset of xtcav23 campaign samples:
full track to DTOTR with the XTCAV off, then checkpointed retracks at +/-V —
the Stage-C reference for the flow->segment->screen image loop.

Standalone by design (run_sweep stays untouched): re-tracks each selected sample
from L0AFEND using the knobs + fixed_overrides recorded in its sample json.
Note these are fresh tracks — the ISR RNG stream differs from the campaign's, so
particle sets won't match the campaign's saved XTCAVF beams exactly; comparisons
are image/observable-level (that is what Stage C needs).

Run under bmad-qpad-dev from the xtcav-gpsr-tuning repo root (long — use tmux):

  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python -u \
    scripts/make_validation_tracks.py --campaign xtcav23_pilot --n 24 [--voltage 22e6]

Outputs (in twobunch-s2e-rl/data/<campaign>_valtracks/):
  sample_XXXXX_XTCAVF_off.h5, sample_XXXXX_DTOTR_{off,plus,minus}.h5
  valtracks_manifest.json

Per sample: one full track (~min at 100k macro) + three ~s retracks from ENDBC20.
Resume-safe: samples with all four beam files present are skipped.
"""

import os

_THREADS = os.environ.get("BMAD_OMP_THREADS", "32")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xtcav_gpsr_tuning import paths

paths.ensure_import_roots()

import FACET2_S2E as qs  # noqa: E402  (bmad-qpad-dev only)

CHECKPOINT = "ENDBC20"
CASES = {"off": (0.0, 0.0), "plus": (+1.0, 0.0), "minus": (-1.0, 0.0)}  # (V sign, phi0 deg)


def pick_indices(sample_jsons: list[Path], n: int) -> list[int]:
    """Evenly-spaced successful, non-baseline-repeat sample indices."""
    ok = []
    for p in sorted(sample_jsons):
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("success") and not d.get("is_baseline_repeat"):
            ok.append(int(d["idx"]))
    if len(ok) <= n:
        return ok
    step = len(ok) / n
    return [ok[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True, help="data/<campaign> under twobunch-s2e-rl")
    ap.add_argument("--n", type=int, default=24, help="number of validation samples")
    ap.add_argument("--indices", type=str, default=None, help="comma list, overrides --n")
    ap.add_argument("--voltage", type=float, default=22e6,
                    help="XTCAV |V| for the plus/minus cases (22 MV = ops readback)")
    ap.add_argument("--nmacro", type=float, default=None,
                    help="override macroparticle count (default: the sample's own)")
    args = ap.parse_args()

    camp_dir = paths.TWOBUNCH_ROOT / "data" / args.campaign
    out_dir = paths.TWOBUNCH_ROOT / "data" / f"{args.campaign}_valtracks"
    scratch = out_dir / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    sample_jsons = sorted(camp_dir.glob("sample_*.json"))
    if not sample_jsons:
        raise SystemExit(f"no sample_*.json under {camp_dir}")
    if args.indices:
        indices = [int(s) for s in args.indices.split(",")]
    else:
        indices = pick_indices(sample_jsons, args.n)
    print(f"[{args.campaign}] {len(indices)} validation samples: {indices}")

    s2e_root = str(paths.FACET2_S2E_ROOT)
    tao = None
    manifest_path = out_dir / "valtracks_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for k, idx in enumerate(indices):
        tag = f"sample_{idx:05d}"
        beams = {case: out_dir / f"{tag}_DTOTR_{case}.h5" for case in CASES}
        beams["xtcavf_off"] = out_dir / f"{tag}_XTCAVF_off.h5"
        if all(p.exists() for p in beams.values()):
            print(f"  [{k+1}/{len(indices)}] {tag}: done, skipping")
            continue

        d = json.loads((camp_dir / f"{tag}.json").read_text())
        merged_base = json.loads(json.dumps(d["knobs"]))  # copy
        fixed = d.get("fixed_overrides", {})
        nmacro = args.nmacro or d["num_macro_particles"]

        if tao is None:
            # baseline_config from the campaign yaml when available (avoids silent
            # divergence if a future campaign changes it); fallback = the current one.
            camp_yaml = paths.TWOBUNCH_ROOT / "configs" / "datagen" / f"{args.campaign}.yaml"
            baseline_rel = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
            if camp_yaml.exists():
                import yaml
                baseline_rel = yaml.safe_load(camp_yaml.read_text()).get(
                    "baseline_config", baseline_rel)
            base_cfg = qs.loadConfig("/" + baseline_rel, s2e_root)
            tao = qs.initializeTao(
                filePath=s2e_root,
                inputBeamFilePathSuffix=base_cfg["inputBeamFilePathSuffix"],
                csrTF=bool(d.get("csrTF", True)),
                transverseWakes=bool(d.get("transverseWakes", True)),
                numMacroParticles=nmacro,
                scratchPath=str(scratch),
                randomizeFileNames=True,
            )
            # makeBeamActiveBeamFile (below) OVERWRITES the active beam file with the
            # ENDBC20 checkpoint; without a restore, every sample after the first would
            # inject a ~10 GeV beam at L0AFEND (the xtcav_sim_scan.py lesson, and critic
            # finding C1). Stash the pristine L0AFEND input and restore it per sample.
            pristine = out_dir / "input_L0AFEND.h5"
            shutil.copy(tao.activeFilePath, pristine)

        shutil.copy(pristine, tao.activeFilePath)  # restore L0AFEND input beam

        # fixed_overrides carries trackEnd: PENT from the campaign — force DTOTR here.
        merged = {**base_cfg, **fixed, **merged_base, "trackEnd": "DTOTR"}
        t0 = time.perf_counter()
        qs.setLattice(tao, **merged)
        tao.cmd("set ele XTCAVF VOLTAGE = 0")
        tao.cmd("set ele XTCAVF PHI0 = 0")
        qs.trackBeam(tao, s2e_root, **merged)
        full_s = time.perf_counter() - t0

        P_ckpt = qs.getBeamAtElement(tao, CHECKPOINT)
        qs.makeBeamActiveBeamFile(P_ckpt, tao=tao)

        rec = {"knobs": d["knobs"], "fixed_overrides": fixed, "voltage_V": args.voltage,
               "num_macro_particles": int(nmacro), "full_track_s": full_s, "cases": {}}
        for case, (vsign, phi_deg) in CASES.items():
            tao.cmd(f"set ele XTCAVF VOLTAGE = {vsign * args.voltage}")
            tao.cmd(f"set ele XTCAVF PHI0 = {phi_deg / 360.0}")
            qs.trackBeam(tao, s2e_root, trackStart=CHECKPOINT, **{**merged, "trackEnd": "DTOTR"})
            P = qs.getBeamAtElement(tao, "DTOTR")
            P.write(str(out_dir / f"{tag}_DTOTR_{case}.h5"))
            rec["cases"][case] = {"n_live_DTOTR": len(P.x)}
            if case == "off":
                Px = qs.getBeamAtElement(tao, "XTCAVF")
                Px.write(str(out_dir / f"{tag}_XTCAVF_off.h5"))
                rec["cases"][case]["n_live_XTCAVF"] = len(Px.x)

        manifest[tag] = rec
        manifest_path.write_text(json.dumps(manifest, indent=1))
        print(f"  [{k+1}/{len(indices)}] {tag}: full {full_s:.0f}s, "
              f"DTOTR live {rec['cases']['plus']['n_live_DTOTR']}", flush=True)

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
