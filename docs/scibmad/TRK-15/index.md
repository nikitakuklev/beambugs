---
title: "TRK-15: SciBmadStandard docstring names a nonexistent Yoshida integrator"
hide:
  - toc
---

# TRK-15 — `SciBmadStandard` docstring names a nonexistent `Yoshida` integrator

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on Beamlines 0.11.0 (`35fe016`, head build) |
| Source | [src/element.jl:250–256](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/element.jl#L250-L256)<br>[ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl:1–32](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl#L1-L32) |

The docstring of the default tracking method `SciBmadStandard` says that elements without an exact map use "the symplectic integrator `Yoshida(order=4, num_steps=1)`". No `Yoshida` type exists in SciBmad, Beamlines or BeamTracking, and no tracking method has a `num_steps` field. The default method maps to `Symplectic` with keyword `n_steps`, at order 2 for exact bodies and order 4 otherwise. A user who copies the documented method gets an `UndefVarError`; tracking itself is unaffected.

Run `julia TRK-15.jl`.

??? example "Reproducer: TRK-15.jl"

    [Download TRK-15.jl](TRK-15.jl)

    ```julia
    # TRK-15: `SciBmadStandard` docstring names a nonexistent `Yoshida` integrator
    #
    # The docstring of the default tracking method `SciBmadStandard` says that elements without an exact
    # map use "the symplectic integrator `Yoshida(order=4, num_steps=1)`". No `Yoshida` type exists in
    # SciBmad, Beamlines or BeamTracking, and no tracking method has a `num_steps` field. The default
    # method maps to `Symplectic` with keyword `n_steps`, at order 2 for exact bodies and order 4
    # otherwise. A user who copies the documented method gets an `UndefVarError`; tracking itself is
    # unaffected.
    #
    # The script reads the `SciBmadStandard` docstring of the loaded Beamlines and checks whether
    # `Yoshida` is defined in SciBmad, Beamlines or BeamTracking. It reports the defect while the
    # docstring names `Yoshida` and no such name is defined, so its verdict reflects the build it runs on.
    using SciBmad
    const BL = SciBmad.Beamlines
    const BT = SciBmad.BeamTracking

    doc = replace(string(@doc BL.SciBmadStandard), r"\s+" => " ")
    doc_yoshida = occursin("Yoshida(order=4, num_steps=1)", doc)
    println("SciBmadStandard docstring names Yoshida(order=4, num_steps=1): ", doc_yoshida)
    defined = isdefined(SciBmad, :Yoshida) || isdefined(BL, :Yoshida) || isdefined(BT, :Yoshida)
    println("Yoshida defined in SciBmad, Beamlines or BeamTracking: ", defined)
    # SciBmadStandard maps to Symplectic with keyword n_steps (order 2 for exact bodies, 4 otherwise).
    println("Symplectic fields: ", fieldnames(BT.Symplectic))
    println(doc_yoshida && !defined ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |

