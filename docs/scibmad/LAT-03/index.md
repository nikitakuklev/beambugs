---
title: "LAT-03: Beamlines R_to_v omits the charge state; ion harmonic RF frequency is superluminal"
hide:
  - toc
---

# LAT-03 — Beamlines `R_to_v` omits the charge state; ion harmonic RF frequency is superluminal

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-07 on Beamlines release [v0.11.0](https://github.com/bmad-sim/Beamlines.jl/tree/v0.11.0) (`35fe016`) |
| Source | [`Beamlines.jl/src/utils.jl:8`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/utils.jl#L8)<br>[`Beamlines.jl/ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl:13`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl#L13)<br>[`Beamlines.jl/src/virtual.jl:557`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L557)<br>[`Beamlines.jl/src/utils.jl:5`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/utils.jl#L5)<br>[`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/utils_bl.jl:205`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L205)<br>[`BeamTracking.jl/src/utils/energy.jl:9`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/utils/energy.jl#L9) |

Beamlines' `R_to_v`, which converts a reference rigidity R to a speed, mishandles the charge state q: it computes |q|·c/√(1 + (m/Rc)²) instead of c/√(1 + (m/(qRc))²) = c·pc/E. The `harmon` ↔ `rf_frequency` getter of an RF element uses this speed, so it is wrong for multiply charged ions; the correct speed agrees with BeamTracking's helpers and the drift reference clock. For ⁴He²⁺ at a reference momentum of 10 GeV/c in a 10 m line with `harmon` = 1, the helper returns 4.807×10⁸ m/s, above c, instead of 2.809×10⁸ m/s, and the getter returns 48.07 MHz instead of 28.09 MHz. Singly charged particles are unaffected, and tracking uses the correct frequency.

Run `julia LAT-03.jl`.

??? example "Reproducer: LAT-03.jl"

    [Download LAT-03.jl](LAT-03.jl)

    ```julia
    # LAT-03: Beamlines `R_to_v` omits the charge state; ion harmonic RF frequency is superluminal
    #
    # Beamlines' `R_to_v`, which converts a reference rigidity R to a speed, mishandles the charge state
    # q: it computes |q|·c/√(1 + (m/Rc)²) instead of c/√(1 + (m/(qRc))²) = c·pc/E. The `harmon` ↔
    # `rf_frequency` getter of an RF element uses this speed, so it is wrong for multiply charged ions;
    # the correct speed agrees with BeamTracking's helpers and the drift reference clock. For ⁴He²⁺ at a
    # reference momentum of 10 GeV/c in a 10 m line with `harmon` = 1, the helper returns 4.807×10⁸ m/s,
    # above c, instead of 2.809×10⁸ m/s, and the getter returns 48.07 MHz instead of 28.09 MHz. Singly
    # charged particles are unaffected, and tracking uses the correct frequency.
    #
    # The script builds a 10 m `RFCavity` Beamline with `harmon` = 1 for ⁴He²⁺ at `pc_ref` = 10 GeV/c.
    # It compares the speed from Beamlines' `R_to_v` and the `rf_frequency` getter with v = c·pc/E and
    # h·v/L. As controls it prints BeamTracking's `R_to_v`, the RF frequency of the tracking setup
    # `rf_omega_calc` and BeamTracking's `R_to_gamma` against E/m.
    using SciBmad
    const BT = SciBmad.BeamTracking
    ext = Base.get_extension(BT, :BeamTrackingBeamlinesExt)
    he = Species("#4He++")
    q = chargeof(he); m = massof(he)
    pc = 10e9; L = 10.0
    bl = Beamline([RFCavity(L=L, harmon=1, voltage=0.0)], species_ref=he, pc_ref=pc)
    R = bl.p_over_q_ref
    # With pc = q R c, v = c pc/sqrt(pc^2 + m^2) = c/sqrt(1 + (m/(q R c))^2); the Beamlines helper
    # computes |q| c/sqrt(1 + (m/(R c))^2), correct only for |q| = 1.
    v_exp = C_LIGHT * pc / sqrt(pc^2 + m^2)
    println("species charge = ", q, ", R = ", R, " T m, pc_ref = ", bl.pc_ref, ", E_ref = ", bl.E_ref)
    v_bl = Beamlines.R_to_v(he, R)
    println("Beamlines.R_to_v   = ", v_bl, " m/s")
    println("expected v = c*pc/sqrt(pc^2+m^2) = ", v_exp, " m/s (c = ", C_LIGHT, ")")
    f_obs = bl.line[1].rf_frequency
    println("rf_frequency getter = ", f_obs, " Hz; expected h*v/L = ", v_exp / L, " Hz")
    w_obs = ext.rf_omega_calc(bl.line[1].RFParams, bl.line[1].BeamlineParams)
    println("controls: BeamTracking.R_to_v = ", BT.R_to_v(he, R), " m/s; tracking omega/2pi = ", w_obs / (2pi),
            " Hz; BeamTracking.R_to_gamma = ", BT.R_to_gamma(he, R), ", E/m = ", bl.E_ref / m)
    bad = !isapprox(v_bl, v_exp; rtol=1e-14) || !isapprox(f_obs, v_exp / L; rtol=1e-14)
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

