"""Stage-C MAP fit harness (plan C2).

Forward model per scan step s and streak sign ±:
    knobs_rec(s) + δ  →  flow (CRN frozen z per bunch)  →  cloud_to_beam
    →  Cheetah segment (δ_tr on Q2/Q1/Q0FF k1; TDC at ±V_eff, per-sample φ_s)
    →  render_two_bunch (single kde call, ν_y y-scale, configured anchor)

Objective:
    L = mean_{s,±} MAE(img, target) + w_q·|split − split_target| + λ·Σ_j (δ_j/σ_j)²

Conventions (critic-reviewed):
* δ parameterized in PHYSICAL units, tanh-bounded to ±bound_frac of the knob box
  FULL span (delta_phys = delta_norm*(hi-lo)); prior σ_j in physical units from the config (TODO-flagged drafts).
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
from .screen import (ScreenGrid, com_center_image, image_zlobe_split,
                     render_two_bunch)
from .segment import (back_drift, build_segment_from_spec, element_by_name,
                      load_spec, numerical_rmatrix)

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
    # PRISTINE δ_tr quad strengths, captured once at load time. ForwardModel
    # writes q.k1 = k1_base*(1+dtr) onto the SHARED segment during every
    # forward, so reading the baseline off the segment at construction made
    # each new model inherit the previous fit's quads — δ_tr compounded
    # multiplicatively across fits in a process (found 2026-08-07: the v2
    # ensemble's +44% Q0FF was accumulated, not fitted).
    k1_pristine: list | None = None


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
    extra = float(cfg.get("dtotr2_extra_drift_m", 0.0) or 0.0)
    if extra > 0:
        # Stage-D: the real DTOTR2 sits downstream of the model's DTOTR
        # (d_nom anchor; CALIBRATION_INVESTIGATION) — append the fixed drift
        import cheetah
        d = cheetah.Drift(length=torch.tensor(extra, dtype=torch.float64,
                                              device=torch.device(device)),
                          name="DTOTR2_CORR")
        d.tracking_method = "drift_kick_drift"
        segment = cheetah.Segment(elements=[segment, d], name=segment.name + "_ext")
    grid = ScreenGrid(bandwidth_m=cfg["bandwidth_um"] * 1e-6, device=device)

    keys = norm["knob_keys"]
    fit_knobs = cfg["fit_knobs"]
    idx = [keys.index(k) for k in fit_knobs]
    lo = np.array(norm["knob_low"]); hi = np.array(norm["knob_high"])
    span = (hi - lo)[idx]
    # float() casts guard against PyYAML's scientific-notation string trap (0.6e6)
    if cfg.get("voltage_V") != "calibrate":  # Stage-D: resolved via calibrate_veff
        cfg["voltage_V"] = float(cfg["voltage_V"])
    cfg["lambda_grid"] = [float(v) for v in cfg["lambda_grid"]]
    sigma_norm = torch.tensor(
        [float(cfg["sigma_phys"][k]) / float(s) for k, s in zip(fit_knobs, span)],
        dtype=torch.float64, device=device)
    bound_norm = torch.tensor(
        [cfg.get("delta_bound_frac", 0.45)] * len(fit_knobs),
        dtype=torch.float64, device=device)
    setup = FitSetup(flow=flow, segment=segment, grid=grid, spec=spec, norm=norm,
                     fit_knobs=fit_knobs, knob_index=idx,
                     sigma_norm=sigma_norm, bound_norm=bound_norm, device=device)
    setup.k1_pristine = [element_by_name(segment, n).k1.detach().clone()
                         for n in setup.dtr_names]
    return cfg, setup


def scale_bunch(cloud: torch.Tensor, log_sdelta: torch.Tensor,
                log_ex: torch.Tensor) -> torch.Tensor:
    """v2 physics nuisances applied to ONE bunch's cloud (..., N, 6) in
    (x, y, z, px, py, pz): scale the pz spread by exp(log_sdelta) (projected
    energy-spread deficit — the y-span gap is δ-content, not a y-scale: the
    thin-direction slice thickness matches real exactly at low compression)
    and the (x, px) spread by exp(log_ex) (emittance scale), each about THIS
    bunch's own centroid (critic: a global two-bunch centroid would scale the
    inter-bunch separation/tilt — an orbit property degenerate with movers)."""
    m = cloud.mean(dim=-2, keepdim=True)
    sx = torch.exp(log_ex)
    sd = torch.exp(log_sdelta)
    one = torch.ones_like(sx)
    s = torch.stack([sx, one, one, sx, one, sd])
    return m + (cloud - m) * s


class ForwardModel(torch.nn.Module):
    """δ, δ_tr, V_eff, ν_y → stack of (steps × signs) unit-sum images + charge split."""

    def __init__(self, setup: FitSetup, base_knobs_norm: torch.Tensor,
                 voltage_V: float, phi_deg: float = 0.0, n_particles: int = 1024,
                 anchor: str = "com", fit_veff: bool = False, seed: int = 0,
                 image_anchor: bool = False, l2ref_nuisance: bool = False,
                 phi_ensemble: list | None = None,
                 base_off_norm: torch.Tensor | None = None,
                 fit_sdelta: bool = False, fit_ex: bool = False):
        """Stage-D extensions (all default-off, Stage-C behavior unchanged):
        image_anchor — render anchor='none' then COM-center the IMAGE (the shared
          Stage-D anchor op; particle-space anchors don't exist on the real side);
        l2ref_nuisance — one fitted scalar shift [deg] on the L2PhaseSet column
          (tight prior applied in map_loss);
        phi_ensemble — list of TDC phase offsets [deg] to average images over
          (forward-modeled residual jitter);
        base_off_norm — (W,23) conditioning for the TCAV-off loss channel."""
        super().__init__()
        self.s = setup
        self.register_buffer("base", base_knobs_norm.to(setup.device))  # (B,23)
        self.n = n_particles
        self.anchor = anchor
        self.phi = phi_deg / 360.0
        self.image_anchor = image_anchor
        self.phi_ensemble = phi_ensemble
        if base_off_norm is not None:
            self.register_buffer("base_off", base_off_norm.to(setup.device))
        else:
            self.base_off = None
        keys = setup.norm["knob_keys"]
        self._l2_col = keys.index("L2PhaseSet")
        self._l2_span_deg = float(setup.norm["knob_high"][self._l2_col]
                                  - setup.norm["knob_low"][self._l2_col])
        self.l2_shift_deg = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float64), requires_grad=l2ref_nuisance)
        g = torch.Generator().manual_seed(seed)
        ld = setup.flow.latent_dim
        self.register_buffer("zd", torch.randn(n_particles, ld, generator=g))
        self.register_buffer("zw", torch.randn(n_particles, ld, generator=g))

        nfit = len(setup.fit_knobs)
        self.delta_raw = torch.nn.Parameter(torch.zeros(nfit, dtype=torch.float64))
        self.dtr = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))  # frac k1
        self.log_nu_y = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        # v2 per-bunch physics nuisances (scale_bunch); default-off = v1 behavior
        self.log_sdelta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64),
                                             requires_grad=fit_sdelta)
        self.log_ex = torch.nn.Parameter(torch.zeros((), dtype=torch.float64),
                                         requires_grad=fit_ex)
        # V_eff as a RELATIVE factor: an absolute volts parameter (2.2e7) cannot be
        # moved by Adam (step bounded by lr ~0.04 V/iter — found via gate (vii)
        # failing with V_eff frozen at exactly its init in every fit)
        self.register_buffer("v_nominal", torch.tensor(float(voltage_V),
                                                       dtype=torch.float64))
        self.veff_frac = torch.nn.Parameter(torch.zeros((), dtype=torch.float64),
                                            requires_grad=fit_veff)

        self._tdc = element_by_name(setup.segment, "XTCAVF")
        self._quads = [element_by_name(setup.segment, n) for n in setup.dtr_names]
        # ALWAYS the pristine strengths — never whatever a previous model left
        # on the shared segment (see FitSetup.k1_pristine).
        if setup.k1_pristine is None:
            setup.k1_pristine = [q.k1.detach().clone() for q in self._quads]
        self._k1_base = [k.detach().clone() for k in setup.k1_pristine]
        for q, k in zip(self._quads, self._k1_base):
            q.k1 = k.clone()  # leave the shared segment in its pristine state
        norm = setup.norm
        self.q_full = (float(norm["drive_full_charge_nC"]) * 1e-9,
                       float(norm["witness_full_charge_nC"]) * 1e-9)
        self.E_ref = float(setup.spec["meta"]["ref_energy_eV"])
        # HANDOFF-PLANE CORRECTION (audit 2026-08-07, critical). The xtcav23
        # campaign saves clouds at the XTCAVF element EXIT ("save_beams_at:
        # [XTCAVF]", V=0 in the campaign, so exit == entrance + an exact
        # tdc_length drift), while the fit segment STARTS at the XTCAVF
        # entrance (it contains the TDC). The exit cloud must therefore be
        # back-drifted by one TDC length before tracking, or the beam receives
        # 1.0334 m of extra transport: R34 is 31% wrong (−4.435 exit vs −3.068
        # entrance; R33·L + R34_exit = R34_entrance closes exactly), R12 only
        # 2.1%, i.e. a y-plane-dominated error. Documented as the intended
        # convention in configs/datagen/xtcav23_pilot.yaml and in the plan, but
        # never implemented in the forward model until now.
        self.handoff_back_drift_m = float(setup.spec["meta"]["tdc_length_m"])

    @property
    def delta_norm(self) -> torch.Tensor:
        return self.s.bound_norm * torch.tanh(self.delta_raw)

    def knobs(self, base: torch.Tensor | None = None) -> torch.Tensor:
        k = (self.base if base is None else base).clone().to(torch.float32)
        d = self.delta_norm.to(torch.float32)
        k[:, self.s.knob_index] = (k[:, self.s.knob_index] + d).clamp(0.0, 1.0)
        if self.l2_shift_deg.requires_grad or float(self.l2_shift_deg) != 0.0:
            k[:, self._l2_col] = (k[:, self._l2_col]
                                  + (self.l2_shift_deg / self._l2_span_deg
                                     ).to(torch.float32)).clamp(0.0, 1.0)
        return k

    def forward(self, rows: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """rows: optional step-row indices into base — used by run_fit_chunked
        to bound the autograd graph (66-image full-batch OOMs at 80 GB)."""
        s = self.s
        knobs = self.knobs()
        if rows is not None:
            knobs = knobs[rows]
        B = knobs.shape[0]
        drive = s.flow.sample_bunch(knobs, 0, self.n, z=self.zd.to(knobs)).double()
        witness = s.flow.sample_bunch(knobs, 1, self.n, z=self.zw.to(knobs)).double()
        drive = scale_bunch(drive, self.log_sdelta, self.log_ex)
        witness = scale_bunch(witness, self.log_sdelta, self.log_ex)
        _, t_d, t_w = s.flow.feasibility(knobs)
        t_d, t_w = t_d.double(), t_w.double()

        for i, (q, k1b) in enumerate(zip(self._quads, self._k1_base)):
            q.k1 = k1b * (1.0 + self.dtr[i])

        images, splits = [], []
        nu_y = torch.exp(self.log_nu_y)
        qd = (t_d * self.q_full[0] / self.n).unsqueeze(-1).expand(B, self.n)
        qw = (t_w * self.q_full[1] / self.n).unsqueeze(-1).expand(B, self.n)
        cloud = torch.cat([drive, witness], dim=1)
        q = torch.cat([qd, qw], dim=1)
        beam = cloud_to_beam(cloud, self.E_ref, q, device=knobs.device)
        beam = back_drift(beam, self.handoff_back_drift_m)  # exit -> entrance
        phis = self.phi_ensemble if self.phi_ensemble else [self.phi * 360.0]
        render_anchor = "none" if self.image_anchor else self.anchor
        for sign in (+1.0, -1.0):
            acc = None
            for phi_deg in phis:
                self._tdc.voltage = sign * self.v_nominal * (1.0 + self.veff_frac)
                self._tdc.phase = torch.tensor(phi_deg / 360.0, dtype=torch.float64,
                                               device=knobs.device)
                out = self.s.segment.track(beam)
                x, y = out.x, out.y
                nd = self.n
                img = render_two_bunch((x[:, :nd], y[:, :nd]),
                                       (x[:, nd:], y[:, nd:]),
                                       qd, qw, s.grid, anchor=render_anchor,
                                       y_scale=nu_y)
                if self.image_anchor:
                    # anchor per phi-member THEN average (mirrors per-shot
                    # anchoring before averaging on the target side); the
                    # zero-padded shift can push charge off-frame, so restore
                    # unit sum to match the target convention (audit #10)
                    img = com_center_image(img)
                    img = img / img.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
                acc = img if acc is None else acc + img
            img_mean = acc / len(phis)
            images.append(img_mean)
            # SAME-ESTIMATOR split (audit #5): measured off the rendered image
            # exactly as the target is, hence sign-dependent. The old flow
            # drive-charge fraction was sign-invariant and unsatisfiable.
            splits.append(image_zlobe_split(img_mean))
        return torch.stack(images), torch.stack(splits)  # (2,B,ny,nx), (2,B)

    def forward_full(self, rows: list | None = None):
        """Streaked (both signs) AND TCAV-off images from ONE flow sample —
        the off channel shares this chunk's conditioning rows (base_off must
        equal base), so resampling the flow for it (forward_off) doubles the
        dominant cost for nothing. Returns (imgs (2,B,ny,nx), splits (2,B),
        off (B,ny,nx))."""
        s = self.s
        assert self.base_off is not None
        knobs = self.knobs()
        if rows is not None:
            knobs = knobs[rows]
        B = knobs.shape[0]
        drive = s.flow.sample_bunch(knobs, 0, self.n, z=self.zd.to(knobs)).double()
        witness = s.flow.sample_bunch(knobs, 1, self.n, z=self.zw.to(knobs)).double()
        drive = scale_bunch(drive, self.log_sdelta, self.log_ex)
        witness = scale_bunch(witness, self.log_sdelta, self.log_ex)
        _, t_d, t_w = s.flow.feasibility(knobs)
        t_d, t_w = t_d.double(), t_w.double()
        for i, (q, k1b) in enumerate(zip(self._quads, self._k1_base)):
            q.k1 = k1b * (1.0 + self.dtr[i])
        nu_y = torch.exp(self.log_nu_y)
        qd = (t_d * self.q_full[0] / self.n).unsqueeze(-1).expand(B, self.n)
        qw = (t_w * self.q_full[1] / self.n).unsqueeze(-1).expand(B, self.n)
        cloud = torch.cat([drive, witness], dim=1)
        q = torch.cat([qd, qw], dim=1)
        beam = cloud_to_beam(cloud, self.E_ref, q, device=knobs.device)
        beam = back_drift(beam, self.handoff_back_drift_m)  # exit -> entrance
        render_anchor = "none" if self.image_anchor else self.anchor
        nd = self.n
        outs = []
        for sign in (+1.0, -1.0, 0.0):
            self._tdc.voltage = sign * self.v_nominal * (1.0 + self.veff_frac)
            self._tdc.phase = torch.tensor(self.phi, dtype=torch.float64,
                                           device=knobs.device)
            out = self.s.segment.track(beam)
            img = render_two_bunch((out.x[:, :nd], out.y[:, :nd]),
                                   (out.x[:, nd:], out.y[:, nd:]),
                                   qd, qw, s.grid, anchor=render_anchor,
                                   y_scale=nu_y)
            if self.image_anchor:
                img = com_center_image(img)
                img = img / img.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
            outs.append(img)
        # same-estimator, sign-dependent split (audit #5)
        splits = torch.stack([image_zlobe_split(outs[0]), image_zlobe_split(outs[1])])
        return (torch.stack(outs[:2]), splits, outs[2])

    def forward_off(self, rows: list | None = None) -> torch.Tensor:
        """TCAV-off loss channel: images at base_off conditioning, V=0.
        Returns (W, ny, nx), COM-centered when image_anchor is set."""
        assert self.base_off is not None, "construct with base_off_norm to use the off channel"
        s = self.s
        knobs = self.knobs(base=self.base_off)
        if rows is not None:
            knobs = knobs[rows]
        W = knobs.shape[0]
        drive = s.flow.sample_bunch(knobs, 0, self.n, z=self.zd.to(knobs)).double()
        witness = s.flow.sample_bunch(knobs, 1, self.n, z=self.zw.to(knobs)).double()
        drive = scale_bunch(drive, self.log_sdelta, self.log_ex)
        witness = scale_bunch(witness, self.log_sdelta, self.log_ex)
        _, t_d, t_w = s.flow.feasibility(knobs)
        t_d, t_w = t_d.double(), t_w.double()
        for i, (qd_, k1b) in enumerate(zip(self._quads, self._k1_base)):
            qd_.k1 = k1b * (1.0 + self.dtr[i])
        self._tdc.voltage = torch.zeros((), dtype=torch.float64, device=knobs.device)
        qd = (t_d * self.q_full[0] / self.n).unsqueeze(-1).expand(W, self.n)
        qw = (t_w * self.q_full[1] / self.n).unsqueeze(-1).expand(W, self.n)
        cloud = torch.cat([drive, witness], dim=1)
        beam = cloud_to_beam(cloud, self.E_ref, torch.cat([qd, qw], dim=1),
                             device=knobs.device)
        beam = back_drift(beam, self.handoff_back_drift_m)  # exit -> entrance
        out = self.s.segment.track(beam)
        nd = self.n
        img = render_two_bunch((out.x[:, :nd], out.y[:, :nd]),
                               (out.x[:, nd:], out.y[:, nd:]),
                               qd, qw, s.grid,
                               anchor="none" if self.image_anchor else self.anchor,
                               y_scale=torch.exp(self.log_nu_y))
        if self.image_anchor:
            img = com_center_image(img)
            img = img / img.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
        return img


def map_loss(model: ForwardModel, targets: torch.Tensor, target_splits: torch.Tensor,
             lam: float, w_q: float = 0.1, mask: torch.Tensor | None = None,
             step_weights: torch.Tensor | None = None,
             targets_off: torch.Tensor | None = None, w_off: float = 0.0,
             prior: str = "ridge", student_nu: float = 4.0,
             l2ref_sigma_deg: float = 0.3, nu_y_sigma_ln: float | None = None,
             sdelta_sigma_ln: float | None = None,
             ex_sigma_ln: float | None = None,
             dtr_sigma_frac: float | None = None) -> dict:
    imgs, splits = model()
    diff = (imgs - targets).abs()
    if mask is not None:
        diff = diff * mask
    per_img = diff.sum(dim=(-2, -1))  # (2,B) TV distances (unit-sum images)
    if step_weights is not None:
        per_img = per_img * step_weights
    l_img = per_img.mean()
    l_split = (splits - target_splits).abs().mean()
    pulls = model.delta_norm / model.s.sigma_norm
    if prior == "ridge":
        l_prior = (pulls ** 2).sum()
    elif prior == "student_t":  # heavy tails: ridge-like for small pulls, slab for real faults
        l_prior = ((student_nu + 1.0) / 2.0
                   * torch.log1p(pulls ** 2 / student_nu)).sum() * 2.0
    else:
        raise ValueError(prior)
    l_nuis = _nuisance_prior(model, l2ref_sigma_deg, nu_y_sigma_ln,
                             sdelta_sigma_ln, ex_sigma_ln, dtr_sigma_frac)
    total = l_img + w_q * l_split + lam * (l_prior + l_nuis)
    l_off = torch.zeros((), dtype=imgs.dtype, device=imgs.device)
    if targets_off is not None and w_off > 0:
        off = model.forward_off()
        l_off = (off - targets_off).abs().sum(dim=(-2, -1)).mean()
        total = total + w_off * l_off
    return {"total": total, "img": l_img.detach(), "split": l_split.detach(),
            "prior": l_prior.detach(), "off": l_off.detach()}


def run_fit(model: ForwardModel, targets, target_splits, lam, iters=150, lr=0.03,
            w_q=0.1, verbose=False, **loss_kwargs):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters,
                                                       eta_min=lr * 0.05)
    hist = []
    for i in range(iters):
        opt.zero_grad()
        parts = map_loss(model, targets, target_splits, lam, w_q, **loss_kwargs)
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
            "sdelta": float(torch.exp(model.log_sdelta)),
            "ex": float(torch.exp(model.log_ex)),
            "veff": float(model.v_nominal * (1.0 + model.veff_frac))}


def _nuisance_prior(model, l2ref_sigma_deg=0.3, nu_y_sigma_ln=None,
                    sdelta_sigma_ln=None, ex_sigma_ln=None,
                    dtr_sigma_frac=None):
    """Nuisance priors in the SAME χ²-style units as the knob ridge; the caller
    multiplies the whole thing by λ (see run_fit_chunked / map_loss).

    v2 made these unit-weight, reasoning that λ-scaling would "loosen a tight
    σ=3% ν_y prior to σ/√λ ≈ 55%". That was wrong: the KNOB ridge is λ-scaled
    too, so λ sets the global data-vs-prior tradeoff, and pulling the nuisances
    out of it made them 1/λ ≈ 100x stronger than any knob prior. Consequence
    (v3, caught by the D0.2 control): every nuisance froze at its prior — σ_δ
    1.018, ν_y 1.0003, L2_ref −0.003° of a 0.3° prior, δ_tr ~1e-5 of a 0.5%
    prior, identical across all 8 restarts — and the pre-registered σ_δ recovery
    test returned 1.04 for a truth of 2.0. Under λ these σ's mean what they
    should: a deviation of σ costs exactly as much as a knob at 1σ_j."""
    l = torch.zeros((), dtype=torch.float64, device=model.base.device)
    if model.l2_shift_deg.requires_grad:
        l = l + (model.l2_shift_deg / l2ref_sigma_deg) ** 2
    if nu_y_sigma_ln is not None:
        l = l + (model.log_nu_y / nu_y_sigma_ln) ** 2
    if sdelta_sigma_ln is not None and model.log_sdelta.requires_grad:
        l = l + (model.log_sdelta / sdelta_sigma_ln) ** 2
    if ex_sigma_ln is not None and model.log_ex.requires_grad:
        l = l + (model.log_ex / ex_sigma_ln) ** 2
    if dtr_sigma_frac is not None:
        # δ_tr was the ONE unregularized DOF. Harmless under v1's 264 µm blur
        # (quads had no leverage, dtr stayed <0.1%), but at the 1-px kernel it
        # is a free 3-DOF width knob — and Q2/Q1/Q0FF are the same magnet class
        # as Q5/Q4/Q3FF, whose σ_j is 0.5% of setting. Same prior, same units.
        l = l + ((model.dtr / dtr_sigma_frac) ** 2).sum()
    return l


def run_fit_chunked(model: ForwardModel, targets, target_splits, lam,
                    iters=150, lr=0.03, w_q=0.1, chunk=8,
                    mask=None, step_weights=None, targets_off=None, w_off=0.0,
                    prior="ridge", student_nu=4.0, l2ref_sigma_deg=0.3,
                    nu_y_sigma_ln=None, sdelta_sigma_ln=None, ex_sigma_ln=None,
                    dtr_sigma_frac=None, off_step_weights=None, verbose=False):
    """run_fit with gradient accumulation over step chunks (Stage D: the 66-image
    + off-channel graph OOMs an 80 GB GPU in one backward; per-chunk backwards
    keep peak memory ~ chunk/B of the full graph). Loss identical to map_loss:
    per-image TVs enter a weighted MEAN over all 2B images (chunk losses are
    scaled by chunk_len/B before backward), the split term likewise, the off
    term a mean over W off images, priors added once."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters,
                                                       eta_min=lr * 0.05)
    B = model.base.shape[0]
    W = model.base_off.shape[0] if (targets_off is not None and w_off > 0) else 0
    hist = []
    for i in range(iters):
        opt.zero_grad()
        tot = 0.0
        combined = W == B  # off channel rides the SAME flow sample (forward_full)
        for lo in range(0, B, chunk):
            rows = list(range(lo, min(lo + chunk, B)))
            if combined:
                imgs, splits, off = model.forward_full(rows=rows)
            else:
                imgs, splits = model(rows=rows)
            diff = (imgs - targets[:, rows]).abs()
            if mask is not None:
                diff = diff * mask
            per_img = diff.sum(dim=(-2, -1))
            if step_weights is not None:
                per_img = per_img * step_weights[:, rows]
            l = per_img.sum() / (2 * B)
            l = l + w_q * (splits - target_splits[:, rows]).abs().sum() / (2 * B)
            if combined:
                d_off = (off - targets_off[rows]).abs().sum(dim=(-2, -1))
                if off_step_weights is not None:
                    d_off = d_off * off_step_weights[rows]
                l = l + w_off * d_off.sum() / W
            l.backward()
            tot += float(l)
        if not combined:
            for lo in range(0, W, chunk):
                rows = list(range(lo, min(lo + chunk, W)))
                off = model.forward_off(rows=rows)
                d_off = (off - targets_off[rows]).abs().sum(dim=(-2, -1))
                if off_step_weights is not None:
                    d_off = d_off * off_step_weights[rows]
                l = w_off * d_off.sum() / W
                l.backward()
                tot += float(l)
        pulls = model.delta_norm / model.s.sigma_norm
        if prior == "ridge":
            l_prior = (pulls ** 2).sum()
        elif prior == "student_t":
            l_prior = ((student_nu + 1.0) / 2.0
                       * torch.log1p(pulls ** 2 / student_nu)).sum() * 2.0
        else:
            raise ValueError(prior)
        l_nuis = _nuisance_prior(model, l2ref_sigma_deg, nu_y_sigma_ln,
                                 sdelta_sigma_ln, ex_sigma_ln, dtr_sigma_frac)
        pen = lam * (l_prior + l_nuis)
        pen.backward()
        tot += float(pen)
        opt.step()
        sched.step()
        hist.append(tot)
        if verbose and i % 25 == 0:
            print(f"  it {i:4d}: total {tot:.5e}", flush=True)
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
            "sdelta": float(torch.exp(model.log_sdelta)),
            "ex": float(torch.exp(model.log_ex)),
            "veff": float(model.v_nominal * (1.0 + model.veff_frac))}


def eval_loss_chunked(model, targets, target_splits, mask=None,
                      step_weights=None, targets_off=None, w_off=0.0,
                      off_step_weights=None, chunk=8) -> dict:
    """No-grad evaluation of the chunked-loss components + per-image TVs."""
    B = model.base.shape[0]
    W = model.base_off.shape[0] if (targets_off is not None and w_off > 0) else 0
    tvs, l_split, l_off = [], 0.0, 0.0
    combined = W == B
    with torch.no_grad():
        for lo in range(0, B, chunk):
            rows = list(range(lo, min(lo + chunk, B)))
            if combined:
                imgs, splits, off = model.forward_full(rows=rows)
                l_off += float((off - targets_off[rows]).abs(
                    ).sum(dim=(-2, -1)).sum())
            else:
                imgs, splits = model(rows=rows)
            diff = (imgs - targets[:, rows]).abs()
            if mask is not None:
                diff = diff * mask
            tvs.append(diff.sum(dim=(-2, -1)).cpu())
            l_split += float((splits - target_splits[:, rows]).abs().sum())
        if not combined:
            for lo in range(0, W, chunk):
                rows = list(range(lo, min(lo + chunk, W)))
                off = model.forward_off(rows=rows)
                l_off += float((off - targets_off[rows]).abs(
                    ).sum(dim=(-2, -1)).sum())
    per_img = torch.cat(tvs, dim=1)  # (2, B)
    weighted = per_img * step_weights.cpu() if step_weights is not None else per_img
    return {"per_image_tv": per_img.numpy(), "img": float(weighted.mean()),
            "split": l_split / (2 * B), "off": l_off / max(W, 1)}


def calibrate_veff(setup: FitSetup, cal_scaled: float,
                   v_probe: float = 28e6) -> float:
    """V_eff anchored via the AS-BUILT fit segment (Stage-D critic C1): the V at
    which the segment's numerically-linearized streak calibration |dx_out/dtau_in|
    (R[0][4] of the full DKD map with the TDC POWERED — not the thin-kick analytic,
    which Stage A measured 1–6% off) equals the measured cal_scaled (m-screen per
    m-z, e.g. 14.40 for E300_15673). The segment is used AS CONSTRUCTED (incl. any
    appended DTOTR2 drift) so the geometry factor appears exactly once; the TDC
    kick is linear in V, so one probe linearization suffices. TDC state restored."""
    tdc = element_by_name(setup.segment, "XTCAVF")
    v0, phi0 = tdc.voltage, tdc.phase
    dev = torch.device(setup.device)
    try:
        tdc.voltage = torch.tensor(float(v_probe), dtype=torch.float64, device=dev)
        tdc.phase = torch.tensor(0.0, dtype=torch.float64, device=dev)
        R = numerical_rmatrix(setup.segment, setup.spec["meta"]["ref_energy_eV"],
                              device=dev)
        streak = abs(float(R[0][4]))  # |dx/dtau| = |dx/dz| (tau = −z)
    finally:
        tdc.voltage, tdc.phase = v0, phi0
    return cal_scaled * v_probe / streak
