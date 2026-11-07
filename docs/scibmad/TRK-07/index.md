---
title: "TRK-07: R_to_v and R_to_gamma are wrong for q ≠ ±1"
hide:
  - toc
---

# TRK-07 — `R_to_v` and `R_to_gamma` are wrong for q ≠ ±1

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on BeamTracking release [v0.8.7](https://github.com/bmad-sim/BeamTracking.jl/tree/v0.8.7) (`65e343f`) |
| Issue | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/366]] |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/368]], fix |
| Source | [src/utils/energy.jl:6](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/energy.jl#L6)<br>[src/utils/energy.jl:9](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/energy.jl#L9)<br>[src/utils.jl:8](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/utils.jl#L8)<br>[ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl:13](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl#L13)<br>[ext/BeamTrackingBeamlinesExt/utils_bl.jl:199–208](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L199-L208)<br>[src/virtual.jl:554–563](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L554-L563)<br>[ext/BeamTrackingBeamlinesExt/BeamTrackingBeamlinesExt.jl:4](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/BeamTrackingBeamlinesExt.jl#L4)<br>[src/utils/energy.jl:7](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/energy.jl#L7) |

For multiply charged ions, the rigidity conversions `R_to_gamma` and `R_to_v` return the wrong Lorentz factor γ and speed, which also corrupts harmonic RF frequencies. The relativistic energy–momentum relation and the independent `E_to_v` helper give the correct charge dependence. For ⁴He²⁺ with reference momentum p₀c = 2mc², twice its rest energy, the helpers return γ = 1.414 instead of 2.236 and a speed of 1.414 instead of 0.8944 times light speed, and a cavity at harmonic 10 in a 100.0 m line gets 42.40 MHz instead of 26.81 MHz. Singly charged species and orbital drift speeds are unaffected.

Run `julia TRK-07.jl`.

??? example "Reproducer: TRK-07.jl"

    [Download TRK-07.jl](TRK-07.jl)

    ```julia
    # TRK-07: `R_to_v` and `R_to_gamma` are wrong for q ≠ ±1
    #
    # For multiply charged ions, the rigidity conversions `R_to_gamma` and `R_to_v` return the wrong
    # Lorentz factor γ and speed, which also corrupts harmonic RF frequencies. The relativistic
    # energy–momentum relation and the independent `E_to_v` helper give the correct charge dependence.
    # For ⁴He²⁺ with reference momentum p₀c = 2mc², twice its rest energy, the helpers return γ = 1.414
    # instead of 2.236 and a speed of 1.414 instead of 0.8944 times light speed, and a cavity at
    # harmonic 10 in a 100.0 m line gets 42.40 MHz instead of 26.81 MHz. Singly charged species and
    # orbital drift speeds are unaffected.
    #
    # The script evaluates `R_to_gamma` and `R_to_v` from BeamTracking and `R_to_v` from Beamlines for
    # ⁴He²⁺ with p₀c = 2mc², and compares them with γ = √5 and v/c = 2/√5; `E_to_v` serves as the
    # control. It then builds a `harmon = 10` cavity in a 100 m line and compares its `rf_frequency`
    # with h·v/C, the harmonic number h times the speed over the circumference C. The verdict requires
    # both `R_to_gamma` and the harmonic frequency to be wrong; the other values are printed for
    # information.
    using SciBmad
    const BT = SciBmad.BeamTracking
    const BL = SciBmad.Beamlines

    s = Species("#4He+2"); M = massof(s); q = chargeof(s)
    pc = 2M; R = pc / (C_LIGHT * q)
    println("species ", nameof(s), "  charge = ", q, "  P0c = 2M")
    # P0c = R q c, hence gamma = sqrt(1 + (R q c/M)^2) = sqrt(5) and v = c |R q c|/sqrt((R q c)^2 +
    # M^2), so v/c = 2/sqrt(5).
    println("BeamTracking.R_to_gamma = ", BT.R_to_gamma(s, R), "   expected sqrt(5) = ", sqrt(5.0))
    println("BeamTracking.R_to_v / c = ", BT.R_to_v(s, R) / C_LIGHT, "   expected 2/sqrt(5) = ", 2 / sqrt(5.0))
    println("Beamlines.R_to_v / c    = ", BL.R_to_v(s, R) / C_LIGHT)
    println("control BT.E_to_v / c   = ", BT.E_to_v(s, hypot(pc, M)) / C_LIGHT)

    # Harmonic RF: frequency = h * v / circumference (Beamlines.jl/src/virtual.jl:554-563)
    bl = Beamline([RFCavity(L = 1.0, voltage = 1e3, harmon = 10), Drift(L = 99.0)], species_ref = s, pc_ref = pc)
    f = bl.line[1].rf_frequency
    fexp = 10 * (2 / sqrt(5.0)) * C_LIGHT / 100.0
    println("harmonic rf_frequency   = ", f, " Hz   expected ", fexp, " Hz   ratio ", f / fexp)
    bad = abs(BT.R_to_gamma(s, R) - sqrt(5.0)) > 1e-6 && abs(f / fexp - 1) > 1e-6
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources (Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float64 |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources (Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources (Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources (Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources (Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float64 |

