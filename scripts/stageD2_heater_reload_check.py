"""Stage-D2: the FACET2-S2E heater-path timing artifact, isolated (no heater pulse).

trackBeam(laserHeater=True) tracks to HTRUNDF (mid heater chicane), writes the beam with
writeBeam (OpenPMD_to_Bmad: timeOffset = the bunch's mean t, time = t - mean t) and reloads it.
Bmad rebuilds each particle's phase-space z from `time` alone (z = -beta c time), so the reload
puts the bunch centroid on the reference particle and drops the centroid z the beam had built
up since L0AFEND. Three routes, golden beam, CSR / wakes off, baseline lattice:
  ref  continuous L0AFEND -> BL11356
  A    FACET2-S2E's reload at HTRUNDF
  D    the same reload, written with time_i = -z_i / (beta_i c) (z from Bmad at HTRUNDF)

  OMP_NUM_THREADS=8 /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
      scripts/stageD2_heater_reload_check.py <scratch dir> [n_particles]

2026-10-31, 2e4 particles: Bmad z at HTRUNDF mean +130.7 um (436 fs ahead of the reference);
A vs ref at BL11356 -266 fs / +578 keV (sigma_t 1558 -> 1616 fs); D vs ref 0.000 fs / 0.001 keV.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/rwu4/photoinjector-rl/twobunch-s2e-rl/.worktrees/stageD2_widebox/src")
import FACET2_S2E as qs  # noqa: E402
import OpenPMD_to_Bmad.Update_h5_file as pmd2bmad  # noqa: E402
from FACET2_S2E.simulators.bmad.core import getBeamAtElement, trackBeamHelper  # noqa: E402
from twobunch_s2e_rl.sim.bmad import make_tao  # noqa: E402

ROOT = Path("/home/rwu4/photoinjector-rl/FACET2-S2E")
C = 299792458.0


def summ(P, tag):
    w = np.asarray(P.weight, float)
    t = np.asarray(P.t, float)
    print(f"{tag:22s} <t> {np.average(t, weights=w) * 1e12:14.6f} ps  "
          f"sigma_t {np.sqrt(np.cov(t, aweights=w)) * 1e15:8.2f} fs  "
          f"<E> {np.average(P.energy, weights=w) / 1e6:10.4f} MeV", flush=True)
    return P


def main():
    scr = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    base = qs.loadConfig("/setLattice_configs/2024-10-14_twoBunch_baseline.yml", str(ROOT))
    tao = make_tao(qs, ROOT, base, csr=False, wakes=False, n_particles=n, scratch=scr,
                   saved_at=("BL11356",))
    qs.setLattice(tao, **base)

    def from_start(end):
        tao.cmd(f"set beam_init position_file={tao.activeFilePath}")
        tao.cmd("reinit beam")
        tao.cmd("set beam_init track_start = L0AFEND")
        tao.cmd(f"set beam_init track_end = {end}")
        trackBeamHelper(tao)

    from_start("BL11356")
    ref = summ(getBeamAtElement(tao, "BL11356", tToZ=False), "ref @BL11356")
    for route in ("A", "D"):
        from_start("HTRUNDF")
        P0 = getBeamAtElement(tao, "HTRUNDF", tToZ=False)
        z = np.asarray(tao.bunch1("HTRUNDF", coordinate="z"), float)
        z = z[np.asarray(tao.bunch1("HTRUNDF", coordinate="state"), int) == 1]
        if route == "A":
            print(f"Bmad z at HTRUNDF: mean {z.mean() * 1e6:.3f} um ({z.mean() / C * 1e15:.1f} fs ahead "
                  f"of the reference particle)", flush=True)
        f = scr / f"reload_{route}.h5"
        f.unlink(missing_ok=True)
        P0.write(str(f))
        pmd2bmad.OpenPMD_to_Bmad(str(f), tOffset=None if route == "A"
                                 else np.asarray(P0.t, float) + z / (np.asarray(P0.beta, float) * C))
        tao.cmd(f"set beam_init position_file={f}")
        tao.cmd("reinit beam")
        tao.cmd("set beam_init track_start = HTRUNDF")
        tao.cmd("set beam_init track_end = BL11356")
        trackBeamHelper(tao)
        P1 = summ(getBeamAtElement(tao, "BL11356", tToZ=False), f"{route} @BL11356")
        dt = (np.asarray(P1.t) - np.asarray(ref.t)) * 1e15        # Bmad keeps particle order
        dE = (np.asarray(P1.energy) - np.asarray(ref.energy)) / 1e3
        print(f"   {route} vs ref per particle: dt {dt.mean():.3f} +- {dt.std():.3f} fs, "
              f"dE {dE.mean():.3f} +- {dE.std():.3f} keV", flush=True)


if __name__ == "__main__":
    main()
