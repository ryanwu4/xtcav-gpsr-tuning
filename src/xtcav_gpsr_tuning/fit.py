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
    # Stage D2 (CNF bundles). charges: knobs_norm (B, n_knob) -> (q_drive, q_witness)
    # injected charges [nC], (B,) each — knob-dependent for per_sample tables
    # (drive_fraction, total charge), constant for the xtcav23 flow. baseline_norm:
    # (n_knob,) normalized baseline row (u=0.5 everywhere for xtcav23).
    charges: object = None
    baseline_norm: torch.Tensor | None = None
    # optional asymmetric δ bounds (normalized units; ForwardModel.delta_norm)
    bound_lo: torch.Tensor | None = None
    bound_hi: torch.Tensor | None = None

    @property
    def latent_dim(self) -> int:
        return int(self.flow.hparams.get("beam_latent_dim", 0) or 0)


def _load_flow(cfg: dict, device: str):
    """(flow, norm, baseline dict | None). cfg['cnf_bundle'] loads a twobunch_cnf
    bundle (Stage D2); otherwise the xtcav23 checkpoint + norm from twobunch-s2e-rl
    via the frozen legacy class (Stage C/D)."""
    if cfg.get("cnf_bundle"):
        import twobunch_cnf
        m = twobunch_cnf.load_model(cfg["cnf_bundle"], device=device)
        baseline = {k["name"]: k["baseline"] for k in m.meta["knobs"]}
        return m.flow, m.norm, baseline
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow

    flow = TwoBunchFlow.load_from_checkpoint(
        str(paths.TWOBUNCH_ROOT / cfg["checkpoint"]), map_location=device)
    norm = json.loads((paths.TWOBUNCH_ROOT / cfg["norm_json"]).read_text())
    return flow, norm, None


def _charges_fn(norm: dict, device: str):
    from twobunch_cnf.charges import injected_charges_nC

    lo = torch.tensor(norm["knob_low"], dtype=torch.float64, device=device)
    hi = torch.tensor(norm["knob_high"], dtype=torch.float64, device=device)
    return injected_charges_nC(norm, norm["knob_keys"], lo, hi)


def load_setup(config_path: str | Path = DEFAULT_CONFIG, device: str = "cpu") -> tuple:
    cfg = yaml.safe_load(Path(config_path).read_text())
    flow, norm, baseline = _load_flow(cfg, device)
    flow = flow.to(device).eval()
    for p in flow.parameters():
        p.requires_grad_(False)
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
    if baseline is None:
        baseline_norm = torch.full((len(keys),), 0.5, dtype=torch.float64, device=device)
    else:
        baseline_norm = torch.tensor(
            [(float(baseline[k]) - l) / (h - l) for k, l, h in
             zip(keys, norm["knob_low"], norm["knob_high"])],
            dtype=torch.float64, device=device)
    setup = FitSetup(flow=flow, segment=segment, grid=grid, spec=spec, norm=norm,
                     fit_knobs=fit_knobs, knob_index=idx,
                     sigma_norm=sigma_norm, bound_norm=bound_norm, device=device,
                     charges=_charges_fn(norm, device), baseline_norm=baseline_norm)
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


def shift_rows(img: torch.Tensor, shifts_px: torch.Tensor) -> torch.Tensor:
    """Shift each row of (..., ny, nx) images along x by shifts_px (..., ny)
    pixels (positive = content moves to +col), linear interpolation, zero fill.
    The torch twin of the deployed ``xtcav.apply_row_shift`` (output column c
    samples input column c - shift); differentiable w.r.t. the image."""
    nx = img.shape[-1]
    c = torch.arange(nx, dtype=img.dtype, device=img.device)
    src = c - shifts_px.to(img).unsqueeze(-1)
    i0 = torch.floor(src)
    w = src - i0
    i0 = i0.long()

    def take(i):
        valid = (i >= 0) & (i < nx)
        return torch.gather(img, -1, i.clamp(0, nx - 1)) * valid

    return (1.0 - w) * take(i0) + w * take(i0 + 1)


def _row_fits(im: np.ndarray, an, min_signal_frac: float = 0.005):
    """``xtcav.fit_row_distortion`` (default arguments) re-implemented line for line,
    also returning each fitted row's full Gaussian parameters (x0, I0, sigma, y0) for
    the implicit gradient. Returns (x0 per row with NaN = no fit, {row: popt})."""
    H, W = im.shape
    x_pix = np.arange(W, dtype=float)
    cen = np.full(H, np.nan)
    popts = {}
    thr = float(min_signal_frac) * float(np.sum(im)) if np.sum(im) > 0 else 0.0
    for iy in range(H):
        line = im[iy, :].astype(float)
        if not (float(np.sum(line)) > thr):
            continue
        try:
            p0 = (float(np.argmax(line)), float(np.max(line)), float(W / 10.0),
                  float(np.median(line)))
            _, popt = an.fit_gaussian(x_pix, line, p0=p0)
            x0 = float(popt[0])
        except Exception:
            continue
        if 0 < x0 < W:
            cen[iy] = x0
            popts[iy] = popt
    return cen, popts


def _fit_sensitivity(popts: dict, lines: torch.Tensor) -> torch.Tensor:
    """(R, nx) d x0 / d lineout of the least-squares Gaussian fit at its optimum, by
    the implicit function theorem: dθ*/dy = (JᵀJ + Σ_i r_i ∇²f_i)⁻¹ Jᵀ (r = f − y),
    first row. lines: (R, nx) the fitted lineouts (rows of popts, in order)."""
    from torch.func import hessian, jacrev, vmap

    x = torch.arange(lines.shape[-1], dtype=torch.float64, device=lines.device)
    th = torch.as_tensor(np.stack(list(popts.values())), dtype=torch.float64,
                         device=lines.device)

    def f(t, x):
        return t[1] * torch.exp(-0.5 * ((x - t[0]) / t[2]) ** 2) + t[3]

    def g(t, x, r):
        return (r * f(t, x)).sum()

    J = vmap(jacrev(f), in_dims=(0, None))(th, x)                     # (R, nx, 4)
    r = vmap(f, in_dims=(0, None))(th, x) - lines.double()           # (R, nx)
    A = J.transpose(1, 2) @ J + vmap(hessian(g), in_dims=(0, None, 0))(th, x, r)
    return torch.linalg.solve(A, J.transpose(1, 2))[:, 0, :]          # (R, nx)


def _interp_matrix(finite: np.ndarray) -> np.ndarray:
    """(H, H) A with shift = -(A @ x0) + const reproducing compute_distortion_shifts'
    np.interp fill (linear between fitted rows, nearest outside)."""
    H = finite.size
    A = np.zeros((H, H))
    idx = np.where(finite)[0]
    for r in range(H):
        if finite[r]:
            A[r, r] = 1.0
        elif r < idx[0]:
            A[r, idx[0]] = 1.0
        elif r > idx[-1]:
            A[r, idx[-1]] = 1.0
        else:
            j1 = idx[np.searchsorted(idx, r)]
            j0 = idx[np.searchsorted(idx, r) - 1]
            A[r, j0], A[r, j1] = (j1 - r) / (j1 - j0), (r - j0) / (j1 - j0)
    return A


def off_row_shifts(off_imgs: torch.Tensor) -> torch.Tensor:
    """(B, ny, nx) TCAV-off images -> (B, ny) per-row shifts [px] from the SAME
    deployed routines the real targets use (``xtcav.fit_row_distortion`` +
    ``compute_distortion_shifts``, default arguments; NaN -> 0).

    The VALUE is exactly the deployed shift. The GRADIENT is the deployed op's own
    first-order response: each fitted row's Gaussian centre is differentiated
    implicitly (_fit_sensitivity), the unfitted rows follow through the same
    linear interpolation (_interp_matrix), and shift = -(x0 - ref) + const
    (projection-peak reference and ROI offset are piecewise constant). Detaching
    the shifts entirely (the first version) dropped a first-order term — the
    off-image drive-witness offset the correction erases moves with the movers —
    and gave e.g. the wrong sign for S2EL_xOffset vs central finite differences
    (port review 2026-10-29)."""
    from .targets_real import _import_deployed

    xtcav, _ = _import_deployed()
    import analysis as an

    arr = off_imgs.detach().cpu().numpy()
    out = np.zeros(arr.shape[:-1])
    terms = []
    for b, im in enumerate(arr):
        d, popts = _row_fits(im, an)
        if not np.isfinite(d).any():
            continue
        sh = xtcav.compute_distortion_shifts(im, d)["shifts_px"]
        out[b] = np.where(np.isfinite(sh), sh, 0.0)
        terms.append((b, d, popts))
    val = torch.as_tensor(out, dtype=off_imgs.dtype, device=off_imgs.device)
    if not off_imgs.requires_grad or not terms:
        return val
    st = torch.zeros_like(val)
    for b, d, popts in terms:
        rows = list(popts)
        lines = off_imgs[b, rows]
        w = _fit_sensitivity(popts, lines.detach()).to(off_imgs.dtype)   # (R, nx)
        x0_lin = (w * lines).sum(-1)                                      # value-free, grad = w
        A = torch.as_tensor(_interp_matrix(np.isfinite(d))[:, rows], dtype=off_imgs.dtype,
                            device=off_imgs.device)                      # (H, R)
        st = st.index_add(0, torch.tensor([b], device=off_imgs.device),
                          (-(A @ (x0_lin - x0_lin.detach()))).unsqueeze(0))
    return val + st


LATENT_MODES = (None, "mean", "shared", "shared_resid")


class ForwardModel(torch.nn.Module):
    """δ, δ_tr, V_eff, ν_y → stack of (steps × signs) unit-sum images + charge split."""

    def __init__(self, setup: FitSetup, base_knobs_norm: torch.Tensor,
                 voltage_V: float, phi_deg: float = 0.0, n_particles: int = 1024,
                 anchor: str = "com", fit_veff: bool = False, seed: int = 0,
                 image_anchor: bool = False, l2ref_nuisance: bool = False,
                 phi_ensemble: list | None = None,
                 base_off_norm: torch.Tensor | None = None,
                 fit_sdelta: bool = False, fit_ex: bool = False,
                 l1ref_nuisance: bool = False, latent_mode: str | None = None,
                 resid_tau: float = 0.3, row_center: bool = False):
        """Stage-D extensions (all default-off, Stage-C behavior unchanged):
        image_anchor — render anchor='none' then COM-center the IMAGE (the shared
          Stage-D anchor op; particle-space anchors don't exist on the real side);
        l2ref_nuisance — one fitted scalar shift [deg] on the L2PhaseSet column
          (tight prior applied in map_loss);
        phi_ensemble — list of TDC phase offsets [deg] to average images over
          (forward-modeled residual jitter);
        base_off_norm — (W,n_knob) conditioning for the TCAV-off loss channel.

        Stage-D2 extensions (CNF bundles; all default-off):
        l1ref_nuisance — one fitted scalar shift [deg] on the L1PhaseSet column
          (the BC11 BLEN feedback re-phases L1 when the injector moves);
        latent_mode — the beam latent of a beam-latent flow, REQUIRED for one
          (left unset, sample_bunch would redraw it from the global RNG on every
          call and the loss would no longer be deterministic in the parameters):
          'mean' = the prior mean of p(s|knobs) at each step's knobs; 'shared' =
          one fitted latent per bunch for the whole scan; 'shared_resid' =
          shared + a per-step residual with prior width resid_tau × the prior σ;
        row_center — the deployed off-shot row-distortion correction on the
          MODEL images, from the model's own TCAV-off image at each step (the
          real targets always carry it, from the real off shots)."""
        super().__init__()
        self.s = setup
        self.register_buffer("base", base_knobs_norm.to(setup.device))  # (B,n_knob)
        self.n = n_particles
        self.anchor = anchor
        self.phi = phi_deg / 360.0
        self.image_anchor = image_anchor
        self.phi_ensemble = phi_ensemble
        self.row_center = row_center
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
        self._l1_col = keys.index("L1PhaseSet")
        self._l1_span_deg = float(setup.norm["knob_high"][self._l1_col]
                                  - setup.norm["knob_low"][self._l1_col])
        self.l1_shift_deg = torch.nn.Parameter(
            torch.zeros((), dtype=torch.float64), requires_grad=l1ref_nuisance)
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

        # beam latent (Stage D2)
        if latent_mode not in LATENT_MODES:
            raise ValueError(f"latent_mode {latent_mode!r} not in {LATENT_MODES}")
        dl = setup.latent_dim
        if dl and latent_mode is None:
            raise ValueError("beam-latent flow: pass latent_mode explicitly "
                             "(None would redraw the latent from the global RNG)")
        if not dl and latent_mode is not None:
            raise ValueError(f"latent_mode={latent_mode!r} but the flow has no beam latent")
        self.latent_mode = latent_mode
        self.resid_tau = float(resid_tau)
        if latent_mode in ("shared", "shared_resid"):
            # Parameterized in PRIOR-σ units about the scan-averaged prior mean:
            # s = s0 + σ̄ ⊙ s_lat, with s0 / σ̄ the prior mean / σ averaged over the
            # scan at δ = 0 (fixed buffers). The witness prior σ is ~0.04-0.07, so
            # an absolute-units parameter moved ~1σ per Adam step at lr 0.04 and
            # restart jitter of 0.3 was 2-5σ (port review). s_lat = 0 starts at
            # the scan-averaged prior mean (not the 'mean' arm, which follows each
            # step's own prior mean).
            with torch.no_grad():
                kb = self.base.to(torch.float32)
                pp = [setup.flow.latent_prior_params(kb, k) for k in (0, 1)]
                s0 = torch.stack([mp.mean(0) for mp, _ in pp]).double()
                sig = torch.stack([torch.exp(0.5 * lp).mean(0) for _, lp in pp]).double()
            self.register_buffer("s0", s0)                           # (2, dl)
            self.register_buffer("s_sig", sig)                       # (2, dl)
            self.s_lat = torch.nn.Parameter(torch.zeros_like(s0))   # (2, dl), σ units
        if latent_mode == "shared_resid":
            if base_off_norm is not None and not torch.equal(base_off_norm, base_knobs_norm):
                # forward_off indexes the residuals with base_off rows (port review #7)
                raise ValueError("shared_resid needs base_off == base")
            self.s_resid = torch.nn.Parameter(                       # σ units
                torch.zeros(2, self.base.shape[0], dl, dtype=torch.float64))

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
        # per_sample tables: T is a fraction of THAT row's injected charge, which
        # moves with drive_fraction / total charge (twobunch_cnf.charges)
        self._per_sample = norm.get("transmission") == "per_sample"
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
        # never implemented in the forward model until now. (C_B3 is trained at
        # the same exit plane, so the same back-drift applies.)
        self.handoff_back_drift_m = float(setup.spec["meta"]["tdc_length_m"])

    @property
    def delta_norm(self) -> torch.Tensor:
        t = torch.tanh(self.delta_raw)
        lo = getattr(self.s, "bound_lo", None)
        if lo is None:  # symmetric bound (Stage C/D)
            return self.s.bound_norm * t
        # Stage D2: per-knob [-lo, +hi] so u = base + δ spans exactly the box
        # (drive_fraction's baseline is u=0.643; a symmetric bound clamped at u=1
        # with zero gradient and never reached its lower edge — port review #4)
        return torch.where(t >= 0, t * self.s.bound_hi, t * lo)

    def knobs(self, base: torch.Tensor | None = None) -> torch.Tensor:
        k = (self.base if base is None else base).clone().to(torch.float32)
        d = self.delta_norm.to(torch.float32)
        k[:, self.s.knob_index] = (k[:, self.s.knob_index] + d).clamp(0.0, 1.0)
        if self.l2_shift_deg.requires_grad or float(self.l2_shift_deg) != 0.0:
            k[:, self._l2_col] = (k[:, self._l2_col]
                                  + (self.l2_shift_deg / self._l2_span_deg
                                     ).to(torch.float32)).clamp(0.0, 1.0)
        if self.l1_shift_deg.requires_grad or float(self.l1_shift_deg) != 0.0:
            k[:, self._l1_col] = (k[:, self._l1_col]
                                  + (self.l1_shift_deg / self._l1_span_deg
                                     ).to(torch.float32)).clamp(0.0, 1.0)
        return k

    # ---- beam latent ---------------------------------------------------------
    def _latent(self, knobs: torch.Tensor, k: int, rows: list | None) -> torch.Tensor | None:
        """(B, dl) latent for bunch k at these (row-selected) knobs, or None."""
        mode = self.latent_mode
        if mode is None:
            return None
        if mode == "mean":
            return self.s.flow.latent_prior_params(knobs, k)[0]
        s = (self.s0[k] + self.s_sig[k] * self.s_lat[k]).unsqueeze(0).expand(knobs.shape[0], -1)
        if mode == "shared_resid":
            r = self.s_resid[k] if rows is None else self.s_resid[k][rows]
            s = s + self.s_sig[k] * r
        return s.to(knobs)

    def latent_chi2(self) -> torch.Tensor:
        """−2 log p(s|knobs) up to a constant, in the knob ridge's χ² units (the
        caller multiplies by λ): for 'shared', the mean over scan steps of
        Σ_d [(s_d − μ_d)²/σ_d² + log σ_d²] under each step's knob-conditioned
        prior (the log-variance term depends on the knobs, so it stays — port
        review #5); for 'shared_resid' the same on s + r_i, plus
        Σ_i Σ_d (σ̄ r_id)²/(τ σ_id)² (each residual is its own parameter).
        Zero for 'mean' / no latent."""
        z = torch.zeros((), dtype=torch.float64, device=self.base.device)
        if self.latent_mode not in ("shared", "shared_resid"):
            return z
        knobs = self.knobs()
        for k in (0, 1):
            mp, lp = self.s.flow.latent_prior_params(knobs, k)
            mp, var = mp.double(), torch.exp(lp.double())
            s = self._latent(knobs, k, None).double()
            z = z + ((s - mp) ** 2 / var + lp.double()).sum(-1).mean()
            if self.latent_mode == "shared_resid":
                r = self.s_sig[k] * self.s_resid[k]
                z = z + (r ** 2 / (self.resid_tau ** 2 * var)).sum()
        return z

    def latent_state(self) -> dict:
        if self.latent_mode not in ("shared", "shared_resid"):
            return {}
        with torch.no_grad():
            knobs = self.knobs()
            out = {}
            for k, name in ((0, "drive"), (1, "witness")):
                mp, lp = self.s.flow.latent_prior_params(knobs, k)
                s = self._latent(knobs, k, None)
                pull = ((s - mp) / torch.exp(0.5 * lp)).double()
                out[name] = {"s": (self.s0[k] + self.s_sig[k] * self.s_lat[k]).tolist(),
                             "s_lat_sigma_units": self.s_lat[k].tolist(),
                             "pull_rms_over_steps": pull.pow(2).mean(0).sqrt().tolist(),
                             "chi2_mean": float(pull.pow(2).sum(-1).mean())}
                if self.latent_mode == "shared_resid":
                    out[name]["resid_rms_sigma_units"] = float(self.s_resid[k].pow(2).mean().sqrt())
            return out

    # ---- forward ---------------------------------------------------------------
    def _beam(self, knobs: torch.Tensor, rows: list | None):
        """flow -> back-drifted cheetah beam at the XTCAVF entrance, plus the
        per-particle charges (B, n) of each bunch."""
        s = self.s
        B = knobs.shape[0]
        kw = [{} if self.latent_mode is None else {"s": self._latent(knobs, k, rows)}
              for k in (0, 1)]
        drive = s.flow.sample_bunch(knobs, 0, self.n, z=self.zd.to(knobs), **kw[0]).double()
        witness = s.flow.sample_bunch(knobs, 1, self.n, z=self.zw.to(knobs), **kw[1]).double()
        drive = scale_bunch(drive, self.log_sdelta, self.log_ex)
        witness = scale_bunch(witness, self.log_sdelta, self.log_ex)
        _, t_d, t_w = s.flow.feasibility(knobs)
        t_d, t_w = t_d.double(), t_w.double()
        for i, (q, k1b) in enumerate(zip(self._quads, self._k1_base)):
            q.k1 = k1b * (1.0 + self.dtr[i])
        if self._per_sample:
            q_d, q_w = s.charges(knobs.double())
            q_d, q_w = q_d.double() * 1e-9, q_w.double() * 1e-9
        else:
            q_d, q_w = self.q_full
        qd = (t_d * q_d / self.n).unsqueeze(-1).expand(B, self.n)
        qw = (t_w * q_w / self.n).unsqueeze(-1).expand(B, self.n)
        cloud = torch.cat([drive, witness], dim=1)
        q = torch.cat([qd, qw], dim=1)
        beam = cloud_to_beam(cloud, self.E_ref, q, device=knobs.device)
        beam = back_drift(beam, self.handoff_back_drift_m)  # exit -> entrance
        return beam, qd, qw

    def _render(self, beam, qd, qw, sign: float, phi_deg: float) -> torch.Tensor:
        """Track at TDC sign×V, phase phi_deg; render (B, ny, nx) un-anchored when
        image_anchor (anchoring is applied after any row correction)."""
        self._tdc.voltage = sign * self.v_nominal * (1.0 + self.veff_frac)
        self._tdc.phase = torch.tensor(phi_deg / 360.0, dtype=torch.float64,
                                       device=qd.device)
        out = self.s.segment.track(beam)
        nd = self.n
        return render_two_bunch((out.x[:, :nd], out.y[:, :nd]),
                                (out.x[:, nd:], out.y[:, nd:]),
                                qd, qw, self.s.grid,
                                anchor="none" if self.image_anchor else self.anchor,
                                y_scale=torch.exp(self.log_nu_y))

    def _anchor(self, img: torch.Tensor) -> torch.Tensor:
        if not self.image_anchor:
            return img
        # the zero-padded shift can push charge off-frame, so restore unit sum
        # to match the target convention (audit #10)
        img = com_center_image(img)
        return img / img.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-30)

    def forward(self, rows: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """rows: optional step-row indices into base — used by run_fit_chunked
        to bound the autograd graph (66-image full-batch OOMs at 80 GB)."""
        knobs = self.knobs()
        if rows is not None:
            knobs = knobs[rows]
        beam, qd, qw = self._beam(knobs, rows)
        shifts = (off_row_shifts(self._render(beam, qd, qw, 0.0, 0.0))
                  if self.row_center else None)
        phis = self.phi_ensemble if self.phi_ensemble else [self.phi * 360.0]
        images, splits = [], []
        for sign in (+1.0, -1.0):
            acc = None
            for phi_deg in phis:
                img = self._render(beam, qd, qw, sign, phi_deg)
                if shifts is not None:
                    img = shift_rows(img, shifts)
                # anchor per phi-member THEN average (mirrors per-shot
                # anchoring before averaging on the target side)
                img = self._anchor(img)
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
        assert self.base_off is not None
        knobs = self.knobs()
        if rows is not None:
            knobs = knobs[rows]
        beam, qd, qw = self._beam(knobs, rows)
        raw = [self._render(beam, qd, qw, sign, self.phi * 360.0)
               for sign in (+1.0, -1.0, 0.0)]
        if self.row_center:
            shifts = off_row_shifts(raw[2])
            raw = [shift_rows(r, shifts) for r in raw]
        outs = [self._anchor(r) for r in raw]
        # same-estimator, sign-dependent split (audit #5)
        splits = torch.stack([image_zlobe_split(outs[0]), image_zlobe_split(outs[1])])
        return (torch.stack(outs[:2]), splits, outs[2])

    def forward_off(self, rows: list | None = None) -> torch.Tensor:
        """TCAV-off loss channel: images at base_off conditioning, V=0.
        Returns (W, ny, nx), COM-centered when image_anchor is set."""
        assert self.base_off is not None, "construct with base_off_norm to use the off channel"
        knobs = self.knobs(base=self.base_off)
        if rows is not None:
            knobs = knobs[rows]
        beam, qd, qw = self._beam(knobs, rows)
        img = self._render(beam, qd, qw, 0.0, self.phi * 360.0)
        if self.row_center:
            img = shift_rows(img, off_row_shifts(img))
        return self._anchor(img)


def map_loss(model: ForwardModel, targets: torch.Tensor, target_splits: torch.Tensor,
             lam: float, w_q: float = 0.1, mask: torch.Tensor | None = None,
             step_weights: torch.Tensor | None = None,
             targets_off: torch.Tensor | None = None, w_off: float = 0.0,
             prior: str = "ridge", student_nu: float = 4.0,
             l2ref_sigma_deg: float = 0.3, nu_y_sigma_ln: float | None = None,
             sdelta_sigma_ln: float | None = None,
             ex_sigma_ln: float | None = None,
             dtr_sigma_frac: float | None = None,
             l1ref_sigma_deg: float = 0.5, lam_cal: float | None = None) -> dict:
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
    if lam_cal is None:
        l_nuis = _nuisance_prior(model, l2ref_sigma_deg, nu_y_sigma_ln,
                                 sdelta_sigma_ln, ex_sigma_ln, dtr_sigma_frac,
                                 l1ref_sigma_deg)
        pen = lam * (l_prior + l_nuis)
    else:  # see run_fit_chunked
        pen = lam * (l_prior + _nuisance_prior(model, l2ref_sigma_deg, None, sdelta_sigma_ln,
                                               ex_sigma_ln, None, l1ref_sigma_deg))
        pen = pen + lam_cal * _nuisance_prior(model, 0.0, nu_y_sigma_ln, None, None,
                                              dtr_sigma_frac, l1ref_sigma_deg, latent=False)
    total = l_img + w_q * l_split + pen
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
            "veff": float(model.v_nominal * (1.0 + model.veff_frac)),
            **_stage_d2_state(model)}


def _stage_d2_state(model) -> dict:
    """Stage-D2 result fields: the L1 shift, the latent state, and the
    box-wall diagnostics of every fitted knob (u_final = its normalized value,
    min/max over scan rows, after the clamp; tanh_sat = |tanh(raw)| — the
    fraction of the δ bound in use)."""
    with torch.no_grad():
        k = model.knobs()
        u = k[:, model.s.knob_index]
        tsat = torch.tanh(model.delta_raw).abs()
    names = model.s.fit_knobs
    u_ph = {c: [float(k[:, j].min()), float(k[:, j].max())]
            for c, j in (("L1PhaseSet", model._l1_col), ("L2PhaseSet", model._l2_col))}
    eps = 5e-3
    walls = [n for i, n in enumerate(names)
             if float(tsat[i]) > 0.95 or float(u[:, i].min()) <= eps or float(u[:, i].max()) >= 1 - eps]
    walls += [c for c, (a, b) in u_ph.items() if a <= eps or b >= 1 - eps]
    return {"l1_shift_deg": float(model.l1_shift_deg),
            "latent": model.latent_state(),
            "u_final": {n: [float(u[:, i].min()), float(u[:, i].max())]
                        for i, n in enumerate(names)},
            "u_phases": u_ph,
            "tanh_sat": dict(zip(names, map(float, tsat))),
            "walls": walls}


def _nuisance_prior(model, l2ref_sigma_deg=0.3, nu_y_sigma_ln=None,
                    sdelta_sigma_ln=None, ex_sigma_ln=None,
                    dtr_sigma_frac=None, l1ref_sigma_deg=0.5, latent=True):
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
    if latent:  # latent=False: the calibration-only terms (ν_y, δ_tr, σ_δ, ε_x)
        if model.l2_shift_deg.requires_grad:
            l = l + (model.l2_shift_deg / l2ref_sigma_deg) ** 2
        if model.l1_shift_deg.requires_grad:
            l = l + (model.l1_shift_deg / l1ref_sigma_deg) ** 2
        # beam latent (Stage D2): the model's own p(s|knobs), same χ² units
        l = l + model.latent_chi2()
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
                    dtr_sigma_frac=None, off_step_weights=None, verbose=False,
                    l1ref_sigma_deg=0.5, lam_cal=None):
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
            d_split = (splits - target_splits[:, rows]).abs()
            if step_weights is not None:  # a zero-weight step (held out / no target) is out
                d_split = d_split * (step_weights[:, rows] > 0)
            l = l + w_q * d_split.sum() / (2 * B)
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
        if lam_cal is None:
            l_nuis = _nuisance_prior(model, l2ref_sigma_deg, nu_y_sigma_ln,
                                     sdelta_sigma_ln, ex_sigma_ln, dtr_sigma_frac,
                                     l1ref_sigma_deg)
            pen = lam * (l_prior + l_nuis)
        else:
            # Stage D2 reachability: the weak ridge λ applies to the CNF's own
            # inputs (knobs, latent, L1/L2 shifts); the measurement-chain
            # calibration terms (ν_y, δ_tr — no box bound) keep their own λ_cal
            l_nuis = _nuisance_prior(model, l2ref_sigma_deg, None, sdelta_sigma_ln,
                                     ex_sigma_ln, None, l1ref_sigma_deg)
            l_cal = _nuisance_prior(model, 0.0, nu_y_sigma_ln, None, None,
                                    dtr_sigma_frac, l1ref_sigma_deg, latent=False)
            pen = lam * (l_prior + l_nuis) + lam_cal * l_cal
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
            "veff": float(model.v_nominal * (1.0 + model.veff_frac)),
            **_stage_d2_state(model)}


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
