---
title: "TRK-15: Stale docstrings: Bunch callback signature, SciBmadStandard \"Yoshida\""
hide:
  - toc
---

# TRK-15 — Stale docstrings: `Bunch` callback signature, `SciBmadStandard` "Yoshida"

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking main (`0bf3585`) |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/370]], fix |
| Source | [src/types.jl:80–86](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/types.jl#L80-L86)<br>[src/element.jl:250–256](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/element.jl#L250-L256)<br>[src/callbacks.jl:4](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/callbacks.jl#L4)<br>[src/callbacks.jl:59–61](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/callbacks.jl#L59-L61)<br>[ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl:1–32](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl#L1-L32) |

Two docstrings describe interfaces that do not exist: the `Bunch` docstring prescribes a three-argument callback that tracking cannot call, and the `SciBmadStandard` docstring names a nonexistent `Yoshida` integrator. The callback call site passes nine arguments, so a callback written as documented, `callback(coords, ds_step, g)`, raises `MethodError`. The default method actually uses `Symplectic` with keyword `n_steps`, at order 2 for exact bodies and order 4 otherwise, not `Yoshida(order=4, num_steps=1)`. Users who write callbacks or configure the documented default integrator explicitly meet these mismatches.

??? example "Reproducer: TRK-15.jl"

    [Download TRK-15.jl](TRK-15.jl)

    ```julia
    # TRK-15: Stale docstrings: `Bunch` callback signature, `SciBmadStandard` "Yoshida"
    #
    # Two docstrings describe interfaces that do not exist: the `Bunch` docstring prescribes a
    # three-argument callback that tracking cannot call, and the `SciBmadStandard` docstring names a
    # nonexistent `Yoshida` integrator. The callback call site passes nine arguments, so a callback
    # written as documented, `callback(coords, ds_step, g)`, raises `MethodError`. The default method
    # actually uses `Symplectic` with keyword `n_steps`, at order 2 for exact bodies and order 4
    # otherwise, not `Yoshida(order=4, num_steps=1)`. Users who write callbacks or configure the
    # documented default integrator explicitly meet these mismatches.
    #
    # The script checks whether the loaded `Bunch` docstring prescribes the three-argument callback,
    # then tracks a drift with a callback of that form and reports the error; separately it checks
    # whether `Yoshida` is defined in SciBmad or BeamTracking. Each part reports the state of the build
    # the script runs on, so where the callback docstring has been corrected the verdict rests on the
    # `Yoshida` part.
    using SciBmad

    const THREE = "callback(coords::Coords, ds_step, g)"
    doc = replace(string(@doc SciBmad.BeamTracking.Bunch), r"\s+" => " ")
    doc_three = occursin(THREE, doc)
    println("(a) Bunch docstring of the loaded BeamTracking prescribes the 3-argument form: ", doc_three)

    # The kernels call every callback as callback(i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref,
    # last_ds_step, last_g, transforms_out!, transforms_in!).
    sp = Species("proton"); pc = 2 * massof(sp)
    documented_cb(coords, ds_step, g) = nothing
    b = Bunch(zeros(1, 6), species = sp, p_over_q_ref = pc / C_LIGHT, callbacks = (documented_cb,))
    ra = try
      track!(b, Beamline([Drift(L = 1.0, tracking_method = Symplectic(n_steps = 2))], species_ref = sp, pc_ref = pc); use_explicit_SIMD = false)
      "no error"
    catch err
      string(typeof(err), ": ", first(first(split(sprint(showerror, err), '\n')), 60), " ...")
    end
    println("(a) 3-argument callback: ", ra)
    # SciBmadStandard maps to Symplectic with keyword n_steps, at order 2 for exact bodies and 4
    # otherwise.
    rb = isdefined(SciBmad, :Yoshida) || isdefined(SciBmad.BeamTracking, :Yoshida)
    println("(b) Yoshida defined in SciBmad or BeamTracking: ", rb)
    defect_a = doc_three && startswith(ra, "MethodError")
    println("(a) reproduced: ", defect_a, "; (b) reproduced: ", !rb)
    println(defect_a || !rb ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found ((a) callback signature) | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Found ((b) Yoshida docstring) | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed ((a) callback signature) | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed ((b) Yoshida docstring) | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |

