---
title: "LAT-03: R_to_v and R_to_gamma omit the charge state; ion harmonic RF frequency is superluminal"
hide:
  - toc
---

# LAT-03 — `R_to_v` and `R_to_gamma` omit the charge state; ion harmonic RF frequency is superluminal

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-29 on Beamlines main (`35fe016`) |
| Issue | [[gh:https://github.com/bmad-sim/BeamTracking.jl/issues/366]] |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/368]], fix |
| Source | [`Beamlines.jl/src/utils.jl:8`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/utils.jl#L8)<br>[`Beamlines.jl/ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl:13`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/ext/BeamlinesGTPSAExt/BeamlinesGTPSAExt.jl#L13)<br>[`BeamTracking.jl/src/utils/energy.jl:9`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/energy.jl#L9)<br>[`BeamTracking.jl/src/utils/energy.jl:6`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/energy.jl#L6)<br>[`Beamlines.jl/src/virtual.jl:557`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L557)<br>[`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/utils_bl.jl:205`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L205)<br>[`Beamlines.jl/src/utils.jl:5`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/utils.jl#L5)<br>[`BeamTracking.jl/src/utils/energy.jl:7`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/utils/energy.jl#L7) |

Beamlines' `R_to_v`, which converts a reference rigidity R to a speed, mishandles the charge state q: it computes |q|·c/√(1 + (m/Rc)²) instead of c/√(1 + (m/(qRc))²) = c·pc/E. A harmonic-number cavity's `rf_frequency` getter uses this speed, so it is wrong for multiply charged ions; the correct speed is consistent with the separately checked drift reference time. For ⁴He²⁺ at a reference momentum of 10 GeV/c in a 10 m line with `harmon` = 1, the helper returns 4.807×10⁸ m/s, above c, instead of 2.809×10⁸ m/s, and the getter returns 48.07 MHz instead of 28.09 MHz. Singly charged particles and the drift reference clock are unaffected.

??? example "Reproducer: LAT-03.jl"

    [Download LAT-03.jl](LAT-03.jl)

    ```julia
    # LAT-03: `R_to_v` and `R_to_gamma` omit the charge state; ion harmonic RF frequency is superluminal
    #
    # Beamlines' `R_to_v`, which converts a reference rigidity R to a speed, mishandles the charge state
    # q: it computes |q|·c/√(1 + (m/Rc)²) instead of c/√(1 + (m/(qRc))²) = c·pc/E. A harmonic-number
    # cavity's `rf_frequency` getter uses this speed, so it is wrong for multiply charged ions; the
    # correct speed is consistent with the separately checked drift reference time. For ⁴He²⁺ at a
    # reference momentum of 10 GeV/c in a 10 m line with `harmon` = 1, the helper returns 4.807×10⁸ m/s,
    # above c, instead of 2.809×10⁸ m/s, and the getter returns 48.07 MHz instead of 28.09 MHz. Singly
    # charged particles and the drift reference clock are unaffected.
    #
    # The script builds a 10 m `RFCavity` Beamline with `harmon` = 1 for ⁴He²⁺ at `pc_ref` = 10 GeV/c.
    # It compares the speed from Beamlines' `R_to_v`, the `rf_frequency` getter and the RF frequency of
    # the tracking setup with v = c·pc/E and h·v/L, and BeamTracking's `R_to_gamma` with E/m.
    # BeamTracking's own `R_to_v` is printed alongside.
    using SciBmad
    const BT = SciBmad.BeamTracking
    ext = Base.get_extension(BT, :BeamTrackingBeamlinesExt)
    he = Species("#4He++")
    q = chargeof(he); m = massof(he)
    pc = 10e9; L = 10.0
    bl = Beamline([RFCavity(L=L, harmon=1, voltage=0.0)], species_ref=he, pc_ref=pc)
    R = bl.p_over_q_ref
    # With pc = q R c, v = c pc/sqrt(pc^2 + m^2) = c/sqrt(1 + (m/(q R c))^2) and gamma = sqrt((q R c/m)^2 + 1);
    # the helpers compute |q| c/sqrt(1 + (m/(R c))^2) and sqrt((R c/m)^2 + 1), correct only for |q| = 1.
    v_exp = C_LIGHT * pc / sqrt(pc^2 + m^2)
    println("species charge = ", q, ", R = ", R, " T m, pc_ref = ", bl.pc_ref, ", E_ref = ", bl.E_ref)
    println("Beamlines.R_to_v   = ", Beamlines.R_to_v(he, R), " m/s")
    println("BeamTracking.R_to_v= ", BT.R_to_v(he, R), " m/s")
    println("expected v = c*pc/sqrt(pc^2+m^2) = ", v_exp, " m/s (c = ", C_LIGHT, ")")
    f_obs = bl.line[1].rf_frequency
    w_obs = ext.rf_omega_calc(bl.line[1].RFParams, bl.line[1].BeamlineParams)
    println("rf_frequency getter = ", f_obs, " Hz; tracking omega/2pi = ", w_obs / (2pi), " Hz; expected h*v/L = ", v_exp / L, " Hz")
    g_obs = BT.R_to_gamma(he, R); g_exp = bl.E_ref / m
    println("BeamTracking.R_to_gamma = ", g_obs, "; expected E/m = ", g_exp)
    bad = !isapprox(f_obs, v_exp / L; rtol=1e-14) || !isapprox(w_obs / (2pi), v_exp / L; rtol=1e-14) ||
          !isapprox(g_obs, g_exp; rtol=1e-14)
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |

