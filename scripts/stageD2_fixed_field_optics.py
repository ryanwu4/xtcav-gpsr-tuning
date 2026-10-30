"""Stage-D2: what the machine-like L2 scan does to the L2 quads when their FIELDS stay fixed (the
real machine during a scan, no re-LEM) instead of their K1 (Bmad/FACET2-S2E default, i.e. an
automatic LEM at every step). No tracking: linear optics only.

Per geometry (l2m_A, l2m_B of probe s2e_l2machine_15673) and scan offset delta from the centre:
setLattice at the scan centre, apply the geometry's centre solution (= the state the machine was
LEM'd in), then apply the step (sim/bmad.l2_machine_solve) with K1 held and with the fields held.
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
from FACET2_S2E.simulators.bmad.lattice.linac import getLinacMatchStrings
from pmd_beamphysics import ParticleGroup
from twobunch_s2e_rl.datagen.paths import facet2_root
from twobunch_s2e_rl.sim.bmad import l2_machine_solve, make_tao, merged_settings

P = Path("/home/rwu4/photoinjector-rl/xtcav-gpsr-tuning/data/s2e_l2machine_15673")
MARK = ["BEGBC14_1", "BEGBC20", "XTCAVF", "DTOTR"]
DELTAS = [-4.0, -2.0, 2.0, 4.0]


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
    group = lambda n: ("fixed" if n.startswith("K11_") else "p4" if n.startswith("K14_4")
                       else "p6" if n.startswith("K14_6") else "scan")
    out = {"beam_twiss_BL11356": T0, "quads": quads, "rows": {}}
    for p in ("l2m_A", "l2m_B"):
        r = next(r for r in rows if r["injector_point"] == p and r["step"] == 0)
        for d in DELTAS:
            qs.setLattice(tao, **merged_settings(base, cfg.get("fixed_overrides", {}), r["knobs"]))
            names = [str(n) for n in getLinacMatchStrings(tao)[1]]
            V = {n: get(n, "VOLTAGE") for n in names}
            G = {n: get(n, "GRADIENT") for n in names}
            phi_m = 360.0 * get(names[0], "PHI0")
            tot = {g: sum(v for n, v in V.items() if group(n) == g) for g in ("fixed", "scan", "p4", "p6")}

            def set_l2(delta):
                sol = l2_machine_solve(tot["fixed"], tot["scan"], (tot["p4"], tot["p6"]), phi_m,
                                       {**r["l2_machine"], "delta_deg": delta})
                ph = {"fixed": sol["phi11_deg"], "scan": sol["psi_deg"],
                      "p4": sol["psi_deg"] - sol["theta_deg"], "p6": sol["psi_deg"] + sol["theta_deg"]}
                for n in names:
                    tao.cmd(f"set ele {n} PHI0 = {ph[group(n)] / 360.0}")
                    tao.cmd(f"set ele {n} GRADIENT = {G[n] * sol['scale']}")

            set_l2(0.0)                          # the state the machine was LEM'd in
            t0, e0 = twiss(), energy()
            set_l2(d)                            # K1 held (S2E default)
            t_k1, e1 = twiss(), energy()
            set_l2(0.0)
            tao.cmd("set ele Quadrupole::* FIELD_MASTER = T")
            set_l2(d)                            # fields held (real machine)
            t_b = twiss()
            tao.cmd("set ele Quadrupole::* FIELD_MASTER = F")
            de = e1 / e0 - 1
            o = {"dE_over_E": dict(zip(quads, de.tolist())),
                 "optics": {m: {pl: {"bmag_fields_held": bmag(t_b[m][pl], t0[m][pl]),
                                     "bmag_K1_held": bmag(t_k1[m][pl], t0[m][pl]),
                                     "beta_ratio_fields_held": t_b[m][pl][0] / t0[m][pl][0]}
                                for pl in "ab"} for m in MARK}}
            out["rows"][f"{p}_{d:+.0f}"] = o
            k = int(np.argmax(np.abs(de)))
            dt = o["optics"]["DTOTR"]
            print(f"{p} delta {d:+.0f}: max |dE/E| {de[k]:+.4f} at {quads[k]}; fields held: Bmag x/y "
                  f"{dt['a']['bmag_fields_held']:.3f}/{dt['b']['bmag_fields_held']:.3f}, beta at DTOTR "
                  f"x{dt['a']['beta_ratio_fields_held']:.2f}/x{dt['b']['beta_ratio_fields_held']:.2f}; "
                  f"K1 held: Bmag max {max(dt[pl]['bmag_K1_held'] for pl in 'ab'):.4f}")
    (P / "fixed_field_optics.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
