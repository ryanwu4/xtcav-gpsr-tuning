"""Stage-D2: what the machine-like L2 scan does to the L2 quads when their FIELDS stay fixed (the
real machine during a scan, no re-LEM) instead of their K1 (Bmad/FACET2-S2E default, i.e. an
automatic LEM at every step). No tracking: linear optics only.

Per geometry (l2m_A, l2m_B of probe s2e_l2machine_15673), LEM point (scan centre or scan start)
and step: the geometry's L2 state at the LEM point (sim/bmad.apply_l2_machine), then the step with
K1 held and with the fields frozen at the LEM point (hold_delta_deg; K1 restored after each).
Reports the reference-energy change at every quad from LI11 to BC14, and Bmag / beta ratios at
BEGBC14_1, BEGBC20, XTCAVF and DTOTR for the tracked beam's own Twiss at BL11356 (90% core,
upstream of L2) propagated with Tao's transfer matrices. Tao's design Twiss is not used: it is
far off the beam in L2 (beta_x ~ 1 km at BC14), which inflates Bmag.

  (bmad-qpad-dev, from the twobunch-s2e-rl stageD2_widebox worktree)
  PYTHONPATH=$PWD/src python /home/rwu4/photoinjector-rl/xtcav-gpsr-tuning/scripts/stageD2_fixed_field_optics.py
-> data/s2e_l2machine_15673/fixed_field_optics.json (+ a printed table)
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import yaml
import FACET2_S2E as qs
from pmd_beamphysics import ParticleGroup
from twobunch_s2e_rl.datagen.paths import facet2_root
from twobunch_s2e_rl.sim.bmad import (apply_l2_machine, make_tao, merged_settings,
                                     release_quad_fields)

P = Path("/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning/data/s2e_l2machine_15673")
MARK = ["BEGBC14_1", "BEGBC20", "XTCAVF", "DTOTR"]
HOLDS = {0.0: [-4.0, -2.0, 2.0, 4.0], -4.0: [-2.0, 0.0, 2.0, 4.0]}   # LEM offset: step offsets (deg from centre)


def bmag(t, t0):
    (b, a), (b0, a0) = t, t0
    return 0.5 * (b0 * (1 + a * a) / b - 2 * a0 * a + (1 + a0 * a0) / b0 * b)


def main():
    cfg = yaml.safe_load((P / "config.yaml").read_text())
    root = facet2_root()
    base = qs.loadConfig("/" + cfg["baseline_config"], str(root))
    tao = make_tao(qs, root, base, csr=False, wakes=False, n_particles=1000,
                   scratch=Path(tempfile.mkdtemp()))
    rows = json.loads((P / "manifest.json").read_text())
    get = lambda n, a: float(tao.lat_list(n, f"ele.{a}", flags="-no_slaves -array_out")[0])

    r0 = next(r for r in rows if r["injector_point"] == "l2m_A" and r["step"] == 0)
    tw = ParticleGroup(str(P / f"sample_{r0['idx']:05d}_BL11356.h5"))
    T0 = {pl: tuple(tw.twiss(plane=x, fraction=0.9)[f"{k}_{x}"] for k in ("beta", "alpha"))
          for pl, x in (("a", "x"), ("b", "y"))}

    def twiss():
        out = {}
        for m in MARK:
            M = np.asarray(tao.matrix("BL11356", m)["mat6"])
            out[m] = {}
            for pl, i in (("a", 0), ("b", 2)):
                R = M[i:i + 2, i:i + 2] / np.sqrt(abs(np.linalg.det(M[i:i + 2, i:i + 2])))
                b0, a0 = T0[pl]
                g0 = (1 + a0 * a0) / b0
                out[m][pl] = (R[0, 0] ** 2 * b0 - 2 * R[0, 0] * R[0, 1] * a0 + R[0, 1] ** 2 * g0,
                              -R[0, 0] * R[1, 0] * b0 + (R[0, 0] * R[1, 1] + R[0, 1] * R[1, 0]) * a0
                              - R[0, 1] * R[1, 1] * g0)
        return out

    lo, hi = get("K11_4A", "s"), get("BEGBC14_1", "s")
    quads = [n for n in dict.fromkeys(str(q) for q in tao.lat_list(
        "Quadrupole::*", "ele.name", flags="-no_slaves -array_out")) if lo < get(n, "s") < hi]
    energy = lambda: np.array([get(n, "E_TOT") for n in quads])
    k1_all = lambda: np.asarray(tao.lat_list("Quadrupole::*", "ele.K1", flags="-no_slaves -array_out"))
    out = {"beam_twiss_BL11356": T0, "quads": quads, "rows": {}}
    for p in ("l2m_A", "l2m_B"):
        r = next(r for r in rows if r["injector_point"] == p and r["step"] == 0)
        lattice = lambda: qs.setLattice(tao, **merged_settings(base, cfg.get("fixed_overrides", {}),
                                                               r["knobs"]))
        for hold, deltas in HOLDS.items():
            for d in deltas:
                spec = {**r["l2_machine"], "delta_deg": hold}
                lattice()
                k1_design = k1_all()
                apply_l2_machine(tao, spec)              # the state the machine was LEM'd in
                t0, e0 = twiss(), energy()
                lattice()
                apply_l2_machine(tao, {**spec, "delta_deg": d})       # K1 held (S2E default)
                t_k1, e1 = twiss(), energy()
                lattice()
                _, held = apply_l2_machine(tao, {**spec, "delta_deg": d, "hold_delta_deg": hold})
                t_b = twiss()                            # fields held (real machine)
                release_quad_fields(tao, held)
                assert np.allclose(k1_all(), k1_design, rtol=1e-12, atol=0), "K1 not restored"
                de = e1 / e0 - 1
                o = {"dE_over_E": dict(zip(quads, de.tolist())),
                     "optics": {m: {pl: {"bmag_fields_held": bmag(t_b[m][pl], t0[m][pl]),
                                         "bmag_K1_held": bmag(t_k1[m][pl], t0[m][pl]),
                                         "beta_ratio_fields_held": t_b[m][pl][0] / t0[m][pl][0]}
                                    for pl in "ab"} for m in MARK}}
                out["rows"][f"{p}_hold{hold:+.0f}_{d:+.0f}"] = o
                k = int(np.argmax(np.abs(de)))
                dt = o["optics"]["DTOTR"]
                print(f"{p} LEM'd at scan {4 + hold:.0f}, step scan {4 + d:.0f}: max |dE/E| {de[k]:+.4f} at "
                      f"{quads[k]}; fields held: Bmag x/y {dt['a']['bmag_fields_held']:.3f}/"
                      f"{dt['b']['bmag_fields_held']:.3f}, beta at DTOTR x{dt['a']['beta_ratio_fields_held']:.2f}"
                      f"/x{dt['b']['beta_ratio_fields_held']:.2f}; K1 held: Bmag max "
                      f"{max(dt[pl]['bmag_K1_held'] for pl in 'ab'):.4f}")
    (P / "fixed_field_optics.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
