"""Stage-D2: Derbenev parameter and overtaking length at every chicane bend (golden baseline, 1D CSR,
Bmad 20250811).

kappa = (sigma_x / sigma_z) sqrt(sigma_x / rho): the 1D CSR model assumes kappa << 1 (Derbenev); FERMI data
(Brynes et al. 2018) found 1D still adequate to kappa ~ 2. L_o = (24 sigma_z rho^2)^(1/3): a bend much
longer than L_o reaches steady state; Bmad's steady_state_3d skips the entrance transient.
sigma_x includes dispersion (all particles of the bunch); evaluated at each bend's exit.

  OMP_NUM_THREADS=8 bmad-qpad-dev/bin/python scripts/stageD2_derbenev.py <scratch dir> [n]
-> <scratch dir>/../derbenev.json (+ a printed table). 2026-11-01, 2e4: BC11 kappa <= 0.04; BC14 bends 2 / 3
0.67 / 1.32; BC20 <= 0.09 (bends split by superimposed elements are not listed); L / L_o 0.1-0.9.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/rwu4/photoinjector-rl/twobunch-s2e-rl/.worktrees/stageD2_widebox/src")
import FACET2_S2E as qs  # noqa: E402
from FACET2_S2E.simulators.bmad.core import trackBeamHelper  # noqa: E402
from twobunch_s2e_rl.sim.bmad import make_tao  # noqa: E402

ROOT = Path("/home/rwu4/photoinjector-rl/FACET2-S2E")
RANGES = {"BC11": "BEGBC11_1:ENDBC11_2", "BC14": "BEGBC14_1:ENDBC14_2", "BC20": "BEGBC20:ENDBC20"}


def main():
    scr = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    base = qs.loadConfig("/setLattice_configs/2024-10-14_twoBunch_baseline.yml", str(ROOT))
    bends = []
    tao = make_tao(qs, ROOT, base, csr=True, wakes=True, n_particles=n, scratch=scr)
    qs.setLattice(tao, **base)
    for chic, rng in RANGES.items():
        ix = tao.lat_list(f"sbend::{rng}", "ele.ix_ele", flags="-no_slaves -track_only")
        for i in ix:
            h = tao.ele_gen_attribs(i)
            if h.get("ANGLE", 0) == 0:
                continue
            bends.append({"chicane": chic, "ix": int(i), "name": tao.ele_head(i)["name"],
                          "L": float(h["L"]), "angle": float(h["ANGLE"]), "rho": float(h["L"] / abs(h["ANGLE"]))})
    tao.cmd("set beam add_saved_at = " + ", ".join(str(b["ix"]) for b in bends))
    tao.cmd(f"set beam_init position_file={tao.activeFilePath}")
    tao.cmd("reinit beam")
    tao.cmd("set beam_init track_start = L0AFEND")
    tao.cmd("set beam_init track_end = ENDBC20")
    trackBeamHelper(tao)
    for b in bends:
        x = np.asarray(tao.bunch1(b["ix"], coordinate="x"), float)
        z = np.asarray(tao.bunch1(b["ix"], coordinate="z"), float)
        st = np.asarray(tao.bunch1(b["ix"], coordinate="state"), int) == 1
        w = np.asarray(tao.bunch1(b["ix"], coordinate="charge"), float)
        drive = st & (w == w[st].max())
        sx, sz = x[drive].std(), z[drive].std()
        b.update(sigma_x_um=sx * 1e6, sigma_z_um=sz * 1e6, kappa=(sx / sz) * np.sqrt(sx / b["rho"]),
                 L_overtake=(24 * sz * b["rho"] ** 2) ** (1 / 3))
        print(f"{b['chicane']} {b['name']:10s} L {b['L']:.3f} m rho {b['rho']:7.2f} m | drive sigma_x "
              f"{b['sigma_x_um']:8.1f} um sigma_z {b['sigma_z_um']:7.1f} um | kappa {b['kappa']:6.2f} | "
              f"L_o {b['L_overtake']:.2f} m (L/L_o {b['L'] / b['L_overtake']:.2f})", flush=True)
    (scr.parent / "derbenev.json").write_text(json.dumps(bends, indent=1))


if __name__ == "__main__":
    main()
