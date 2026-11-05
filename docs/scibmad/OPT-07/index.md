---
title: "OPT-07: With RF, the parametric slip/etac/alphac drop sin(2πφ₃)"
hide:
  - toc
---

# OPT-07 — With RF, the parametric `slip`/`etac`/`alphac` drop sin(2πφ₃)

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Source | [`src/twiss/twiss_operators.jl:621-626`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L621-L626)<br>[`twiss_operators.jl:663-668`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L663-L668)<br>[`twiss_operators.jl:617-618`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L617-L618)<br>[`NonlinearNormalForm.jl/src/sagan_rubin.jl:48-50`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/sagan_rubin.jl#L48-L50) |

With RF on, a `twiss` run with knob parameters and a maximum order above 1 omits the factor sin(2πφ₃), with φ₃ the synchrotron phase advance in turns, from `slip`, `etac` and `alphac`. Adding an unused knob must preserve each constant value, and in a ring without dispersion the tracked one-turn M₅₆/β₀ (β₀ the reference speed divided by c) agrees with the slip computed without knobs. For 1 GeV/c protons in a thin-quadrupole cell with a 2 MV, harmonic-1 cavity, the unused knob changes `slip` from 1.604 m to 17.87 m and `etac` from −0.4678 to −5.214. Order-1 runs with knobs take the separate matrix branch, which keeps the sine.

??? example "Reproducer: OPT-07.jl"

    [Download OPT-07.jl](OPT-07.jl)

    ```julia
    # OPT-07: With RF, the parametric `slip`/`etac`/`alphac` drop sin(2πφ₃)
    #
    # With RF on, a `twiss` run with knob parameters and a maximum order above 1 omits the factor
    # sin(2πφ₃), with φ₃ the synchrotron phase advance in turns, from `slip`, `etac` and `alphac`.
    # Adding an unused knob must preserve each constant value, and in a ring without dispersion the
    # tracked one-turn M₅₆/β₀ (β₀ the reference speed divided by c) agrees with the slip computed
    # without knobs. For 1 GeV/c protons in a thin-quadrupole cell with a 2 MV, harmonic-1 cavity, the
    # unused knob changes `slip` from 1.604 m to 17.87 m and `etac` from −0.4678 to −5.214. Order-1 runs
    # with knobs take the separate matrix branch, which keeps the sine.
    #
    # The script adds a 0.5 m, 2 MV, harmonic-1 cavity to a periodic cell of thin quadrupoles and 1 m
    # drifts (1 GeV/c protons) and runs `twiss` with `Descriptor(6,2)` and with `Descriptor(6,2,1,2)`,
    # which adds one unused knob; the end-of-line `slip` must agree. The ring has no dispersion, so its
    # longitudinal block is decoupled and the ForwardDiff one-turn Jacobian of `track` gives M₅₆ =
    # B₃[5,6]·sin μ₃ exactly: M₅₆/β₀ checks the slip without knobs, and M₅₆/(β₀ sin μ₃) matches the
    # value returned with the knob.
    using SciBmad, LinearAlgebra
    const FD = SciBmad.ForwardDiff
    num(x) = x isa AmplitudeDependentValue ? getterm(x) : x
    sp = Species("proton"); pc = 1e9
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    rf = RFCavity(L = 0.5, voltage = 2e6, harmon = 1, zero_phase = PhaseRef.BelowTransition)
    bl = Beamline([qf, d, qd, d, rf], species_ref = sp, pc_ref = pc)

    # Both representations implement slip = B₃[5,6]·sin(2πφ₃)/β: the matrix B₃ without parameters, the
    # map B₃ with parameters and max order > 1. The map branch returns B₃[5,6]/β without the sine.
    res = Dict()
    for (name, desc) in (("Descriptor(6,2)", Descriptor(6, 2)), ("Descriptor(6,2,1,2) unused knob", Descriptor(6, 2, 1, 2)))
        tw = twiss(bl, GTPSA_descriptor = desc, cols = ["slip", "phi3"])
        s = scalar(tw.slip[end]); p = scalar(tw.phi3[end])
        res[name] = s
        println(name, ": slip(END)=", s, " phi3(END)=", p, " etac=", scalar(num(tw.etac)), " alphac=", scalar(num(tw.alphac)), " q3=", scalar(num(tw.q3)))
    end
    mp = 938.27208943e6; β0 = (pc / mp) / sqrt(1 + (pc / mp)^2)
    # No dispersion, so the longitudinal block is decoupled: M₅₆ = B₃[5,6]·sin μ₃ exactly and slip(END)
    # = M₅₆/β₀.
    M = FD.jacobian(v -> vec(track(bl, v0 = reshape(v, 1, 6)).v[1, :, end]), zeros(6))
    μ = acos((M[5, 5] + M[6, 6]) / 2)
    println("independent: tracked M56/β0=", M[5, 6] / β0, "  μ3/2π=", μ / 2pi,
            "  M56/(β0·sin μ3) = B3[5,6]/β0 = ", M[5, 6] / (β0 * sin(μ * sign(M[5, 6]))),
            "  max|M[1:4,5:6]|=", maximum(abs, M[1:4, 5:6]))
    a, b = values(res)
    println(abs(a - b) > 1e-9 * abs(a) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |

