---
title: "TRK-01: Ordinary RF decelerates at the declared accelerating crest"
hide:
  - toc
---

# TRK-01 — Ordinary RF decelerates at the declared accelerating crest

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P0 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-24 on BeamTracking main (`d8e45fd`) |
| Source | [ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:482](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L482)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:497](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L497)<br>[src/kernels/rfcavity_kick.jl:106–114](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/rfcavity_kick.jl#L106-L114)<br>[ext/BeamTrackingBeamlinesExt/utils_bl.jl:214–226](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L214-L226)<br>[src/rf.jl:4–5](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/rf.jl#L4-L5)<br>[src/rf.jl:66](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/rf.jl#L66)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:484](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L484)<br>[ext/BeamTrackingBeamlinesExt/utils_bl.jl:217–222](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L217-L222) |

Tracking an ordinary `RFCavity` with `SciBmadStandard`, `Symplectic` or `DriftKick` at `phi0 = 0` and the default `zero_phase = PhaseRef.Accelerating` decelerates particles instead of accelerating them. That phase reference is documented as the maximum-accelerating phase, and `SaganCavity` tracking gives the required energy gain. A proton or electron with reference momentum p₀c = 2mc², twice its rest energy, crossing a 1.000 µm, 1.000 kV cavity loses 1.000 keV instead of gaining 1.000 keV. Every linac or energy-gain setup that uses ordinary RF with the default phase reference is affected; `BelowTransition` and `AboveTransition` have the correct phase on this path.

??? example "Reproducer: TRK-01.jl"

    [Download TRK-01.jl](TRK-01.jl)

    ```julia
    # TRK-01: Ordinary RF decelerates at the declared accelerating crest
    #
    # Tracking an ordinary `RFCavity` with `SciBmadStandard`, `Symplectic` or `DriftKick` at `phi0 = 0`
    # and the default `zero_phase = PhaseRef.Accelerating` decelerates particles instead of accelerating
    # them. That phase reference is documented as the maximum-accelerating phase, and `SaganCavity`
    # tracking gives the required energy gain. A proton or electron with reference momentum p₀c = 2mc²,
    # twice its rest energy, crossing a 1.000 µm, 1.000 kV cavity loses 1.000 keV instead of gaining
    # 1.000 keV. Every linac or energy-gain setup that uses ordinary RF with the default phase reference
    # is affected; `BelowTransition` and `AboveTransition` have the correct phase on this path.
    #
    # The script tracks a proton and an electron with p₀c = 2mc², starting on the reference orbit,
    # through a 1000 V, 1 MHz cavity at `phi0 = 0` with `zero_phase = PhaseRef.Accelerating`: 1 µm long
    # with `SciBmadStandard`, and zero length with `SaganCavity` as the control. It compares each energy
    # gain with the required +|q|V = +1000 eV. A recheck shows that `phi0 = π` on the ordinary path
    # gives +1000 eV, so the ordinary phase is off by exactly π.
    using SciBmad
    const BT = SciBmad.BeamTracking

    function energy_gain(species, method)
      M = massof(species); pc = 2M; E0 = hypot(pc, M); V = 1000.0
      ele = RFCavity(L = (method isa SaganCavity ? 0.0 : 1e-6), voltage = V, rf_frequency = 1e6,
                     phi0 = 0.0, zero_phase = PhaseRef.Accelerating, tracking_method = method)
      if method isa SaganCavity     # Sagan reads dE_ref from its Beamline inside a Branch
        br = Branch([Beamline([Marker()], species_ref = species, pc_ref = pc),
                     Beamline([ele], dE_ref = 0.0)])
        bl = br.beamlines[2]
      else
        bl = Beamline([ele], species_ref = species, pc_ref = pc)
      end
      b = Bunch(zeros(1, 6), species = species, p_over_q_ref = pc / (C_LIGHT * chargeof(species)))
      track!(b, bl; use_explicit_SIMD = false)
      return hypot(pc * (1 + b.v[1, 6]), M) - E0, b.state[1]
    end

    # Beamlines defines PhaseRef.Accelerating as the maximum-accelerating phase, so at phi0 = 0 the
    # reference particle must gain +|q|V for either sign of charge.
    bad = false
    for s in (Species("proton"), Species("electron"))
      dE, st = energy_gain(s, SciBmadStandard())
      dEs, sts = energy_gain(s, SaganCavity(n_cells = 0, L_active = 0.0))
      println(rpad(nameof(s), 9), " SciBmadStandard dE = ", dE, " eV  (state ", Int(st), ")")
      println(rpad(nameof(s), 9), " SaganCavity     dE = ", dEs, " eV  (state ", Int(sts), ")")
      println(rpad(nameof(s), 9), " expected        dE = +1000 eV  (|q|V at the declared crest)")
      global bad |= abs(dE - 1000) > 1e-5 && abs(dE + 1000) < 1e-5
    end
    # --- Independent recheck: the ordinary kernel is off by exactly pi at the crest ---------------
    let s = Species("proton"), M = massof(s), pc = 2M
      b = Bunch(zeros(1, 6), species = s, p_over_q_ref = pc / C_LIGHT)
      track!(b, Beamline([RFCavity(L = 1e-6, voltage = 1000.0, rf_frequency = 1e6, phi0 = pi)], species_ref = s, pc_ref = pc); use_explicit_SIMD = false)
      println("recheck: ordinary, Accelerating, phi0 = pi: dE = ", hypot(pc * (1 + b.v[1, 6]), M) - hypot(pc, M), " eV")
    end
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64 |

