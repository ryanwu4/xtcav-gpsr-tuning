"""Stage-C MAP fit harness (plan C2).

Forward model per scan step s and streak sign ±:
    knobs_rec(s) + δ  →  flow (CRN frozen z per bunch)  →  cloud_to_beam
    →  Cheetah segment (δ_tr on Q2/Q1/Q0FF k1; TDC at ±V_eff, per-sample φ_s)
    →  render_two_bunch (single kde call, ν_y y-scale, configured anchor)

Objective:
    L = mean_{s,±} MAE(img, target) + w_q·|split − split_target| + λ·Σ_j (δ_j/σ_j)²

Conventions (critic-reviewed):
* δ parameterized in PHYSICAL units, tanh-bounded to ±bound_frac of the knob box
  half-span; prior σ_j in physical units from the config (TODO-flagged drafts).
* ν_y and V_eff carry NO prior (a bound ν_y would push kicker-type η_y signatures
  into the movers and corrupt the kicker-elimination gate).
* Fit z-buffers are frozen per fit; TARGETS must come from an independent draw
  (self-recovery) or Bmad (truth recovery) — never the fit's own buffer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from . import paths
from .convert import cloud_to_beam
from .screen import ScreenGrid, render_two_bunch
from .segment import build_segment_from_spec, element_by_name, load_spec

paths.ensure_import_roots()

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "fit_stageC.yaml"


@dataclass
class FitSetup:
    flow: object
    segment: object
    grid: ScreenGrid
    spec: dict
    norm: dict
    fit_knobs: list          # names of the 17 δ_up dims
    knob_index: list         # their column indices in the 23-dim frame
    sigma_norm: torch.Tensor  # prior σ per fitted knob, in NORMALIZED knob units
    bound_norm: torch.Tensor  # tanh bound per fitted knob, normalized units
    dtr_names: list = field(default_factory=lambda: ["Q2FF", "Q1FF", "Q0FF"])
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


def load_setup(config_path: str | Path = DEFAULT_CONFIG, device: str = "cpu") -> tuple:
    cfg = yaml.safe_load(Path(config_path).read_text())
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow

    flow = TwoBunchFlow.load_from_checkpoint(
        str(paths.TWOBUNCH_ROOT / cfg["checkpoint"]), map_location=device
    ).to(device).eval()
    for p in flow.parameters():
        p.requires_grad_(False)
    norm = json.loads((paths.TWOBUNCH_ROOT / cfg["norm_json"]).read_text())
    spec = load_spec(paths.REPO_ROOT / cfg["segment_spec"])
    segment, _ = build_segment_from_spec(spec, dtype=torch.float64,
                                         device=torch.device(device))
    grid = ScreenGrid(bandwidth_m=cfg["bandwidth_um"] * 1e-6, device=device)

    keys = norm["knob_keys"]
    fit_knobs = cfg["fit_knobs"]
    idx = [keys.index(k) for k in fit_knobs]
    lo = np.array(norm["knob_low"]); hi = np.array(norm["knob_high"])
    span = (hi - lo)[idx]
    # float() casts guard against PyYAML's scientific-notation string trap (0.6e6)
    cfg["voltage_V"] = float(cfg["voltage_V"])
    cfg["lambda_grid"] = [float(v) for v in cfg["lambda_grid"]]
    sigma_norm = torch.tensor(
        [float(cfg["sigma_phys"][k]) / float(s) for k, s in zip(fit_knobs, span)],
        dtype=torch.float64, device=device)
    bound_norm = torch.tensor(
        [cfg.get("delta_bound_frac", 0.45)] * len(fit_knobs),
        dtype=torch.float64, device=device)
    return cfg, FitSetup(flow=flow, segment=segment, grid=grid, spec=spec, norm=norm,
                         fit_knobs=fit_knobs, knob_index=idx,
                         sigma_norm=sigma_norm, bound_norm=bound_norm, device=device)


class ForwardModel(torch.nn.Module):
    """δ, δ_tr, V_eff, ν_y → stack of (steps × signs) unit-sum images + charge split."""

    def __init__(self, setup: FitSetup, base_knobs_norm: torch.Tensor,
                 voltage_V: float, phi_deg: float = 0.0, n_particles: int = 1024,
                 anchor: str = "com", fit_veff: bool = False, seed: int = 0):
        super().__init__()
        self.s = setup
        self.register_buffer("base", base_knobs_norm.to(setup.device))  # (B,23)
        self.n = n_particles
        self.anchor = anchor
        self.phi = phi_deg / 360.0
        g = torch.Generator().manual_seed(seed)
        ld = setup.flow.latent_dim
        self.register_buffer("zd", torch.randn(n_particles, ld, generator=g))
        self.register_buffer("zw", torch.randn(n_particles, ld, generator=g))

        nfit = len(setup.fit_knobs)
        self.delta_raw = torch.nn.Parameter(torch.zeros(nfit, dtype=torch.float64))
        self.dtr = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))  # frac k1
        self.log_nu_y = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        # V_eff as a RELATIVE factor: an absolute volts parameter (2.2e7) cannot be
        # moved by Adam (step bounded by lr ~0.04 V/iter — found via gate (vii)
        # failing with V_eff frozen at exactly its init in every fit)
        self.register_buffer("v_nominal", torch.tensor(float(voltage_V),
                                                       dtype=torch.float64))
        self.veff_frac = torch.nn.Parameter(torch.zeros((), dtype=torch.float64),
                                            requires_grad=fit_veff)

        self._tdc = element_by_name(setup.segment, "XTCAVF")
        self._quads = [element_by_name(setup.segment, n) for n in setup.dtr_names]
        self._k1_base = [q.k1.detach().clone() for q in self._quads]
        norm = setup.norm
        self.q_full = (float(norm["drive_full_charge_nC"]) * 1e-9,
                       float(norm["witness_full_charge_nC"]) * 1e-9)
        self.E_ref = float(setup.spec["meta"]["ref_energy_eV"])

    @property
    def delta_norm(self) -> torch.Tensor:
        return self.s.bound_norm * torch.tanh(self.delta_raw)

    def knobs(self) -> torch.Tensor:
        k = self.base.clone().to(torch.float32)
        d = self.delta_norm.to(torch.float32)
        k[:, self.s.knob_index] = (k[:, self.s.knob_index] + d).clamp(0.0, 1.0)
        return k

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.s
        knobs = self.knobs()
        B = knobs.shape[0]
        drive = s.flow.sample_bunch(knobs, 0, self.n, z=self.zd.to(knobs)).double()
        witness = s.flow.sample_bunch(knobs, 1, self.n, z=self.zw.to(knobs)).double()
        _, t_d, t_w = s.flow.feasibility(knobs)
        t_d, t_w = t_d.double(), t_w.double()

        for i, (q, k1b) in enumerate(zip(self._quads, self._k1_base)):
            q.k1 = k1b * (1.0 + self.dtr[i])

        images, splits = [], []
        nu_y = torch.exp(self.log_nu_y)
        for sign in (+1.0, -1.0):
            self._tdc.voltage = sign * self.v_nominal * (1.0 + self.veff_frac)
            self._tdc.phase = torch.tensor(self.phi, dtype=torch.float64,
                                           device=knobs.device)
            qd = (t_d * self.q_full[0] / self.n).unsqueeze(-1).expand(B, self.n)
            qw = (t_w * self.q_full[1] / self.n).unsqueeze(-1).expand(B, self.n)
            cloud = torch.cat([drive, witness], dim=1)
            q = torch.cat([qd, qw], dim=1)
            beam = cloud_to_beam(cloud, self.E_ref, q, device=knobs.device)
            out = self.s.segment.track(beam)
            x, y = out.x, out.y
            nd = self.n
            img = render_two_bunch((x[:, :nd], y[:, :nd]), (x[:, nd:], y[:, nd:]),
                                   qd, qw, s.grid, anchor=self.anchor, y_scale=nu_y)
            images.append(img)
            split = (t_d * self.q_full[0]) / (t_d * self.q_full[0] + t_w * self.q_full[1])
            splits.append(split)
        return torch.stack(images), torch.stack(splits)  # (2,B,ny,nx), (2,B)


def map_loss(model: ForwardModel, targets: torch.Tensor, target_splits: torch.Tensor,
             lam: float, w_q: float = 0.1) -> dict:
    imgs, splits = model()
    # per-image L1/total-variation distance (images are unit-sum, so this is
    # scale-free and O(0.01-1)); a mean over ALL pixels would sit at ~1e-6 and
    # be crushed by any sensible λ·prior (found the hard way in the smoke test)
    l_img = (imgs - targets).abs().sum(dim=(-2, -1)).mean()
    l_split = (splits - target_splits).abs().mean()
    pulls = model.delta_norm / model.s.sigma_norm
    l_prior = (pulls ** 2).sum()
    total = l_img + w_q * l_split + lam * l_prior
    return {"total": total, "img": l_img.detach(), "split": l_split.detach(),
            "prior": l_prior.detach()}


def run_fit(model: ForwardModel, targets, target_splits, lam, iters=150, lr=0.03,
            w_q=0.1, verbose=False):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters,
                                                       eta_min=lr * 0.05)
    hist = []
    for i in range(iters):
        opt.zero_grad()
        parts = map_loss(model, targets, target_splits, lam, w_q)
        parts["total"].backward()
        opt.step()
        sched.step()
        hist.append(float(parts["total"]))
        if verbose and i % 25 == 0:
            print(f"  it {i:4d}: total {hist[-1]:.5e} img {float(parts['img']):.5e}")
    with torch.no_grad():
        pulls = (model.delta_norm / model.s.sigma_norm).cpu().numpy()
        span = np.array([model.s.norm["knob_high"][j] - model.s.norm["knob_low"][j]
                         for j in model.s.knob_index])
        delta_phys = model.delta_norm.cpu().numpy() * span
    return {"loss_history": hist,
            "delta_phys": dict(zip(model.s.fit_knobs, map(float, delta_phys))),
            "pulls": dict(zip(model.s.fit_knobs, map(float, pulls))),
            "dtr": model.dtr.detach().cpu().tolist(),
            "nu_y": float(torch.exp(model.log_nu_y)),
            "veff": float(model.v_nominal * (1.0 + model.veff_frac))}
