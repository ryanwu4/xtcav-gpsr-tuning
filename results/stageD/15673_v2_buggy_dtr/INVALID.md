# INVALIDATED — quad k1 state leak (found 2026-08-07)

`ForwardModel` writes `q.k1 = k1_base*(1+dtr)` onto the SHARED Cheetah segment on
every forward, and captured `k1_base` by reading the segment at construction. So
each new model inherited the PREVIOUS fit's quad strengths: **δ_tr compounded
multiplicatively across every fit in a process** (orient 4 → ensemble 16 →
sensitivity 7, in sequence).

Symptoms that exposed it: the ensemble reported δ_tr = −13.6% / −15.1% / **+44%**
k1 (physically absurd), and re-rendering a "δ_tr = 0" variant produced output
bit-identical to the fitted model — impossible unless both were reading the same
mutated quads.

v1 (264 µm kernel) was affected in principle but immune in practice: the blur left
the quads no leverage, so per-fit δ_tr stayed <0.1% and accumulation was negligible.

Fix: `FitSetup.k1_pristine` captured once at load; every `ForwardModel` restores
it. Regression test: `tests/test_targets_real.py::test_no_quad_k1_leak_between_models`.
Also added in the same pass: a δ_tr prior (σ = 0.5% fractional k1, the same
provenance as the Q5/Q4/Q3FF σ_j) — δ_tr had been the one unregularized DOF.

What in here is still valid: `precal/register.json` (built from a single model,
so no accumulation) and the target files. Everything under `fit/` is void.
