---
title: "COL-01: IBS source statistics include lost particles"
hide:
  - toc
---

# COL-01 — IBS source statistics include lost particles

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking main (`0bf3585`, head build) |
| Issue | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/380]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/381]], filed from this evidence |
| Source | [`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/general_bl.jl:115-129`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/general_bl.jl#L115-L129)<br>[`BeamTracking.jl/src/utils/beam_statistics.jl:4-48`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/utils/beam_statistics.jl#L4-L48)<br>[`BeamTracking.jl/src/kernels/ibs_kick.jl:5`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/ibs_kick.jl#L5)<br>[`85-87`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/ibs_kick.jl#L85-L87)<br>[`bmad/multiparticle/beam_utils.f90:1367,1379-1380`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/multiparticle/beam_utils.f90#L1367-L1380) |

The intrabeam scattering (IBS) kick computes the particle count, centroid and covariance that set live-particle kicks over the whole bunch, including particles marked lost. The kernel kicks only live particles, so these statistics must describe the live sample; removing the lost row, or giving it zero weight, gives the same live-only result. For 200 live protons at 1.000 GeV, each representing 1.000 × 10⁴ particles, one lost particle 20 standard deviations out changes the largest deterministic kick through a 1.000 m drift by 66.02% of its live-only value instead of zero. A lost row holding NaN or Inf can make tracking throw `SingularException`.

Run `julia COL-01.jl`.

??? example "Reproducer: COL-01.jl"

    [Download COL-01.jl](COL-01.jl)

    ```julia
    # COL-01: IBS source statistics include lost particles
    #
    # The intrabeam scattering (IBS) kick computes the particle count, centroid and covariance that set
    # live-particle kicks over the whole bunch, including particles marked lost. The kernel kicks only
    # live particles, so these statistics must describe the live sample; removing the lost row, or
    # giving it zero weight, gives the same live-only result. For 200 live protons at 1.000 GeV, each
    # representing 1.000 × 10⁴ particles, one lost particle 20 standard deviations out changes the
    # largest deterministic kick through a 1.000 m drift by 66.02% of its live-only value instead of
    # zero. A lost row holding NaN or Inf can make tracking throw `SingularException`.
    #
    # The script tracks 200 Gaussian-distributed live protons, each representing 10⁴ particles, plus one
    # outlier at 20 standard deviations through a 1 m drift with deterministic IBS damping only, three
    # times: with the outlier marked lost, marked alive and removed. It compares the live particles'
    # momentum changes with the lost outlier against those with the outlier removed, and shows that they
    # equal the result with the outlier alive. As a control, it keeps the lost row but gives it weight
    # zero, which must reproduce the removed-row kicks.
    using SciBmad, Random, LinearAlgebra
    const BT = SciBmad.BeamTracking

    Random.seed!(1)
    n = 200
    sig = [1e-4, 1e-5, 1e-4, 1e-5, 1e-3, 1e-4]
    vlive = randn(n, 6) .* sig'
    outlier = 20 .* sig'                       # a lost particle far outside the core
    sp = Species("proton")
    drift = Drift(L = 1.0, tracking_method = Symplectic(ibs_damping_on = true, ibs_fluctuations_on = false))
    line = Beamline([drift], species_ref = sp, E_ref = 1.0e9)
    const W = 1e4                               # macroparticle weight: N = W * rows

    function run(v, state)
        b = Bunch(v = copy(v), state = copy(state), weight = W, species = sp, p_over_q_ref = line.p_over_q_ref)
        track!(b, line; use_explicit_SIMD = false)
        return b
    end

    va = vcat(vlive, outlier)
    bL = run(va, vcat(fill(BT.STATE_ALIVE, n), BT.STATE_LOST))   # outlier lost
    bA = run(va, fill(BT.STATE_ALIVE, n + 1))                     # outlier alive
    bR = run(vlive, fill(BT.STATE_ALIVE, n))                      # outlier removed

    dp(b) = b.coords.v[1:n, [2, 4, 6]] .- vlive[:, [2, 4, 6]]   # IBS momentum change of live rows
    ref = dp(bR)
    println("outlier row state after tracking = ", bL.coords.state[end], " (STATE_LOST = ", BT.STATE_LOST, ")")
    println("max |dp_live| with outlier removed           = ", maximum(abs, ref))
    println("max |dp_live| with outlier lost (kept in rows) = ", maximum(abs, dp(bL)), "  (|dp|/sigma_p = ", maximum(abs, ref ./ sig[[2,4,6]]'), ")")
    println("max |dp(lost outlier) - dp(removed)| / max|dp| = ", maximum(abs, dp(bL) .- ref) / maximum(abs, ref))
    println("max |dp(lost outlier) - dp(alive outlier)|     = ", maximum(abs, dp(bL) .- dp(bA)))
    # Population moments seen by the builder vs the live sample:
    _, cA = BT.mean_and_cov(va, nothing, BT.CPU())
    _, cR = BT.mean_and_cov(vlive, nothing, BT.CPU())
    println("Sigma_xx all rows = ", cA[1, 1], "  live rows = ", cR[1, 1], "  ratio = ", cA[1, 1] / cR[1, 1])
    # Independent recheck (weighted statistics path): give the lost row weight 0 instead.
    bW = let b = Bunch(v = copy(va), state = vcat(fill(BT.STATE_ALIVE, n), BT.STATE_LOST), weight = vcat(fill(W, n), 0.0),
                       species = sp, p_over_q_ref = line.p_over_q_ref)
        track!(b, line; use_explicit_SIMD = false); b
    end
    println("recheck: weight-0 lost row vs row removed, max rel. difference = ", maximum(abs, dp(bW) .- ref) / maximum(abs, ref))
    rel = maximum(abs, dp(bL) .- ref) / maximum(abs, ref)
    println(rel > 1e-6 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), source, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), source, Julia 1.13.2, CPU, Float64, serial |

