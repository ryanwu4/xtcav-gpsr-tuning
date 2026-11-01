"""Stage-D2: re-track rows of an existing S2E probe with Bmad's 3D steady-state CSR.

Copies the source probe's manifest / reference files into data/<out>, links its cached Impact-T beams
(no Impact rerun), patches twobunch's sim.bmad.make_tao so every Tao it builds (including bmad_task's
retry) switches the chosen ranges to csr_method = steady_state_3d, and runs
injector_campaign.bmad_task per row in this process. Score with score_s2e_probe.py --conv B after.

Ranges: BC11 / BC14 / BC20 = the whole chicane (drifts too: steady_state_3d computes no field in a
drift, so the 1D exit-transient CSR of the drifts is lost); BC11b / BC14b / BC20b = the bends only
(drifts keep 1_Dim). "1d" = reference (no change), "nocsr" = CSR off.

Needs Bmad >= 20261023: steady_state_3d segfaults in track1_bunch_csr3d with bmad-qpad-dev's 20250811
(even on a one-bend toy lattice). That Tao also rejects FACET2-S2E's tao init files (datum lists and
plot templates), so the run goes through a mirror of FACET2-S2E (--mirror, built if missing): symlinks
everywhere except bmad/models/f2_elec, a real directory holding patched init copies, which also keeps
Bmad's digested-lattice cache out of the FACET2-S2E checkout. The 2026-11-01 runs used a scratch env
(bmad 20261023 + pytao 1.3.3 from the local conda cache) with FACET2_S2E, OpenPMD_to_Bmad, twobunch_cnf,
lume-impact and their pure-python deps on PYTHONPATH from bmad-qpad-dev:

  ulimit -s unlimited; FACET2_S2E_ROOT=<mirror> TWOBUNCH_BMAD_THREADS=32 PYTHONPATH=<shim>:<FACET2-S2E>/src \
  <scratch env>/bin/python scripts/stageD2_csr3d_probe.py s2e_csrbin32_15673 s2e_csr3d_allb_hi \
      BC11b,BC14b,BC20b --mirror <mirror> [--idx 0 1] [--mesh 32 32 64]
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
TB = REPO.parent / "twobunch-s2e-rl/.worktrees/stageD2_widebox/src"
S2E = REPO.parent / "FACET2-S2E"
RANGES = {"BC11": "BEGBC11_1:ENDBC11_2", "BC14": "BEGBC14_1:ENDBC14_2", "BC20": "BEGBC20:ENDBC20"}
RANGES.update({k + "b": "sbend::" + v for k, v in list(RANGES.items())})


def build_mirror(dst: Path) -> None:
    """FACET2-S2E as symlinks, except bmad/models/f2_elec: real copies of the two tao init files with
    the &tao_d1_data / &tao_d2_data namelists and the plot_file line commented out (tracking uses no
    Tao data or plots), and no digested files (Bmad writes its cache beside the lattice it opened)."""
    def link_all(src, d, keep):
        d.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.name not in keep and ".digested" not in p.name and not (d / p.name).exists():
                (d / p.name).symlink_to(p)

    link_all(S2E, dst, {"bmad"})
    link_all(S2E / "bmad", dst / "bmad", {"models"})
    link_all(S2E / "bmad/models", dst / "bmad/models", {"f2_elec"})
    f2 = S2E / "bmad/models/f2_elec"
    link_all(f2, dst / "bmad/models/f2_elec", {"tao.init", "tao_transverseWakesOn.init"})
    for name in ("tao.init", "tao_transverseWakesOn.init"):
        out, inside = [], False
        for line in (f2 / name).read_text().splitlines():
            inside = inside or bool(re.match(r"\s*&tao_d[12]_data", line))
            out.append("!" + line if inside or line.strip().startswith("plot_file") else line)
            inside = inside and line.strip() != "/"
        (dst / "bmad/models/f2_elec" / name).write_text("\n".join(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("ranges", help="comma list of RANGES keys, or 1d / nocsr")
    ap.add_argument("--idx", type=int, nargs="*")
    ap.add_argument("--mesh", nargs=3, default=["32", "32", "64"])
    ap.add_argument("--mirror", type=Path, help="FACET2-S2E mirror for the newer Tao (built if missing)")
    a = ap.parse_args()
    if a.mirror:
        if not a.mirror.exists():
            build_mirror(a.mirror)
        os.environ["FACET2_S2E_ROOT"] = str(a.mirror)
    sys.path.insert(0, str(TB))
    src, out = REPO / "data" / a.src, REPO / "data" / a.out
    out.mkdir(parents=True, exist_ok=True)
    for f in ("manifest.json", "manifest_meta.json", "points.json", "l0af_reference.json", "campaign.json"):
        if (src / f).exists() and not (out / f).exists():
            shutil.copy2(src / f, out / f)
    if not (out / "beams").exists():
        (out / "beams").symlink_to(src / "beams")
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    rngs = [] if a.ranges in ("1d", "nocsr") else [RANGES[k] for k in a.ranges.split(",")]
    cfg.pop("csr_n_bin", None)      # the reference probes predate twobunch 77cf183: they ran 40 bins
    if a.ranges == "nocsr":
        cfg["csrTF"] = False
    cfg.update(name=a.out, output_dir=str(out),
               csr3d={"ranges": rngs, "mesh": [int(m) for m in a.mesh], "source_probe": a.src})
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    from twobunch_s2e_rl.datagen import injector_campaign as ic
    from twobunch_s2e_rl.sim import bmad as simb
    from twobunch_s2e_rl.sim import injector as sim_inj

    orig = simb.make_tao

    def make_tao_3d(*args, **kw):
        tao = orig(*args, **kw)
        tao.cmd(f"set space_charge_com csr3d_mesh_size = {' '.join(a.mesh)}")
        for rng in rngs:
            tao.cmd(f"set ele {rng} csr_method = steady_state_3d")
        print(f"[csr3d] steady_state_3d on {rngs}, mesh {a.mesh}", flush=True)
        return tao

    simb.make_tao = make_tao_3d
    ic._init_worker(cfg)
    ref = sim_inj.pin_l0af_reference(out / "l0af_reference.json", out / "l0af_autophase")
    golden = ic.injector_golden(ic.resolve_sweep_set(cfg["injector_sweep_set"])[0])
    ref_info = json.loads((ic._beam_dir(golden, ref) / "injector_info.json").read_text())
    for row in ic.build_manifest(cfg):
        if (a.idx and row["idx"] not in a.idx) or ic._is_done(ic.row_json_path(out, row)):
            continue
        t0 = time.perf_counter()
        idx, ok, _ = ic.bmad_task((row, ref, ref_info))
        print(f"[csr3d] idx {idx} step {row.get('step')} {'ok' if ok else 'FAIL'} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
