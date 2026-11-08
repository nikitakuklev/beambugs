---
title: "TRK-08: Interior callbacks get the entrance t_ref in RF cavities and separators"
hide:
  - toc
---

# TRK-08 — Interior callbacks get the entrance `t_ref` in RF cavities and separators

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [src/utils/find_stuff.jl:22–43](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/utils/find_stuff.jl#L22-L43)<br>[src/kernels/integrators.jl:20–21](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/integrators.jl#L20-L21)<br>[src/kernels/integrators.jl:82–83](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/integrators.jl#L82-L83)<br>[src/callbacks.jl:4](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/callbacks.jl#L4)<br>[src/callbacks.jl:26–29](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/callbacks.jl#L26-L29) |

A user callback inside an ordinary RF cavity or an electrostatic separator receives the element's entrance reference time at every interior step. The callback contract requires the reference time at the current position s, which at fixed reference energy grows with the reference flight time s/(β₀c), where β₀c is the reference speed. For a proton with reference momentum p₀c = 2mc², twice its rest energy, in a 2.000 m element tracked in four second-order symplectic steps, the callback at s = 1.500 m sees a time increment of 0 instead of 5.594 ns. Drifts report the correct times, and the element-exit callback is correct.

Run `julia TRK-08.jl`.

??? example "Reproducer: TRK-08.jl"

    [Download TRK-08.jl](TRK-08.jl)

    ```julia
    # TRK-08: Interior callbacks get the entrance `t_ref` in RF cavities and separators
    #
    # A user callback inside an ordinary RF cavity or an electrostatic separator receives the element's
    # entrance reference time at every interior step. The callback contract requires the reference time
    # at the current position s, which at fixed reference energy grows with the reference flight time
    # s/(β₀c), where β₀c is the reference speed. For a proton with reference momentum p₀c = 2mc², twice
    # its rest energy, in a 2.000 m element tracked in four second-order symplectic steps, the callback
    # at s = 1.500 m sees a time increment of 0 instead of 5.594 ns. Drifts report the correct times,
    # and the element-exit callback is correct.
    #
    # The script tracks a proton with p₀c = 2mc² through 2 m elements with `Symplectic(order=2,
    # n_steps=4)` and a callback that records the reference time it receives: a drift as the control, an
    # `RFCavity`, and an electrostatic separator (`En0`). It compares each recorded time with the
    # entrance time plus s/(β₀c). A recheck calls the reference-clock helper `compute_dt_ref` on
    # parameter tuples shaped like the builders', which returns s/(β₀c) for the drift kernel and 0 for
    # the cavity and separator kernels.
    using SciBmad

    const sp = Species("proton"); const pc = 2 * massof(sp); const beta0 = 2 / sqrt(5)
    const rec = Tuple{Float64,Float64}[]
    cb(i, coords, s, t, bg, ds, g, tout, tin) = (push!(rec, (Float64(s), Float64(t))); nothing)

    function record(ele)
      empty!(rec)
      b = Bunch(zeros(1, 6), species = sp, p_over_q_ref = pc / C_LIGHT, t_ref = 1e-6, callbacks = (cb,))
      track!(b, Beamline([ele], species_ref = sp, pc_ref = pc); use_explicit_SIMD = false)
      return copy(rec)
    end

    # The callback argument cur_t_ref is the reference time at cur_s; the integrators add
    # compute_dt_ref(s) to t_enter, and at fixed reference energy this is t_enter + s/(beta0 c).
    tm = Symplectic(order = 2, n_steps = 4)
    worst = Dict{String,Float64}()
    for (name, ele) in (("drift (control)", Drift(L = 2.0, tracking_method = tm)),
                        ("RF cavity", RFCavity(L = 2.0, voltage = 1e3, rf_frequency = 1e8, tracking_method = tm)),
                        ("E separator", LineElement(L = 2.0, En0 = 1e3, tracking_method = tm)))
      r = record(ele)
      err = maximum(abs(t - (1e-6 + s / (beta0 * C_LIGHT))) for (s, t) in r)
      worst[name] = err
      println(rpad(name, 16), " (s, t_ref - t_enter) = ", [(s, t - 1e-6) for (s, t) in r])
      println(rpad("", 16), " max |t_ref - (t_enter + s/(beta0 c))| = ", err, " s")
    end
    bad = worst["drift (control)"] < 1e-20 && worst["RF cavity"] > 1e-9 && worst["E separator"] > 1e-9
    # --- Independent recheck: the reference-clock helper on builder-shaped parameter tuples -------
    const BT = SciBmad.BeamTracking
    using SciBmad: StaticArrays
    let tm_ = 0.5, b0 = beta0, g2 = 5.0, SA = StaticArrays.SA
      pd = (b0, g2, tm_)
      prf = (nothing, b0, g2, tm_, 0.0, 1.0, 0.0, 1.0, 0, Val{false}(), SA{Int}[], SA{Float64}[], SA{Float64}[])
      pel = (nothing, 1.0, b0, tm_, nothing, nothing, 0.0)
      println("recheck: compute_dt_ref(0.5 m): exact_drift! = ", BT.compute_dt_ref(0.5, BT.exact_drift!, pd),
              ", cavity! = ", BT.compute_dt_ref(0.5, BT.cavity!, prf), ", elsep_rad! = ", BT.compute_dt_ref(0.5, BT.elsep_rad!, pel),
              "; s/(beta0 c) = ", 0.5 / (b0 * C_LIGHT))
    end
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-22</span> | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |

