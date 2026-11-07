---
title: "TRK-02: SaganCavity swaps the below/above-transition zero phases"
hide:
  - toc
---

# TRK-02 — `SaganCavity` swaps the below/above-transition zero phases

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [ext/BeamTrackingBeamlinesExt/sagan_cavity_bl.jl:15](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/sagan_cavity_bl.jl#L15)<br>[src/kernels/sagan_cavity.jl:188–189](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L188-L189)<br>[ext/BeamTrackingBeamlinesExt/utils_bl.jl:217–220](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L217-L220)<br>[src/rf.jl:6–7](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/rf.jl#L6-L7)<br>[src/utils/z_to_time.jl:26](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/utils/z_to_time.jl#L26) |

With `SaganCavity` tracking, the `BelowTransition` and `AboveTransition` zero-phase references select each other's stable zero crossing, so the longitudinal restoring kick has the wrong sign and the cavity runs on the unstable slope. Below transition a late particle must gain energy and above transition it must lose energy, as the phase-stability condition and independent tracking in a drift-only ring establish. In a 1.000 kV, 100.0 MHz cavity, a proton or electron at 0.8944 times light speed arriving late, at longitudinal offset z = −0.02000 m, loses 46.85 eV below transition instead of gaining 46.85 eV. The `Accelerating` crest and ordinary RF transition phases are correct.

Run `julia TRK-02.jl`.

??? example "Reproducer: TRK-02.jl"

    [Download TRK-02.jl](TRK-02.jl)

    ```julia
    # TRK-02: `SaganCavity` swaps the below/above-transition zero phases
    #
    # With `SaganCavity` tracking, the `BelowTransition` and `AboveTransition` zero-phase references
    # select each other's stable zero crossing, so the longitudinal restoring kick has the wrong sign
    # and the cavity runs on the unstable slope. Below transition a late particle must gain energy and
    # above transition it must lose energy, as the phase-stability condition and independent tracking in
    # a drift-only ring establish. In a 1.000 kV, 100.0 MHz cavity, a proton or electron at 0.8944 times
    # light speed arriving late, at longitudinal offset z = −0.02000 m, loses 46.85 eV below transition
    # instead of gaining 46.85 eV. The `Accelerating` crest and ordinary RF transition phases are
    # correct.
    #
    # The script tracks a proton and an electron with p₀c = 2mc², late by z = −0.02 m, through a 1 kV,
    # 100 MHz cavity at `phi0 = 0` under both transition references, with `SaganCavity` and with
    # ordinary RF as the control. It compares each energy change with +|q|V·sin(ωΔt) below transition
    # and −|q|V·sin(ωΔt) above, where Δt is the particle's lateness and ω the angular RF frequency. A
    # recheck follows a proton for 40 turns in a drift-only ring, which is below transition, and prints
    # the growth of its longitudinal amplitude: a correct `BelowTransition` cavity keeps it bounded and
    # a correct `AboveTransition` one does not.
    using SciBmad

    function gain(species, method, zp, z)
      M = massof(species); pc = 2M; E0 = hypot(pc, M)
      ele = RFCavity(L = (method isa SaganCavity ? 0.0 : 1e-6), voltage = 1000.0, rf_frequency = 1e8,
                     phi0 = 0.0, zero_phase = zp, tracking_method = method)
      if method isa SaganCavity
        br = Branch([Beamline([Marker()], species_ref = species, pc_ref = pc), Beamline([ele], dE_ref = 0.0)])
        bl = br.beamlines[2]
      else
        bl = Beamline([ele], species_ref = species, pc_ref = pc)
      end
      v = zeros(1, 6); v[1, 5] = z
      b = Bunch(v, species = species, p_over_q_ref = pc / (C_LIGHT * chargeof(species)))
      track!(b, bl; use_explicit_SIMD = false)
      return hypot(pc * (1 + b.v[1, 6]), M) - E0
    end

    bad = false
    for s in (Species("proton"), Species("electron"))
      M = massof(s); beta = 2 / sqrt(5)              # pc = 2M
      z = -0.02                                      # late particle
      dt = -z / (beta * C_LIGHT); w = 2pi * 1e8
      # z = beta*c*(t_ref - t), so z < 0 is late. phi0 = 0 is the stable zero crossing: below transition
      # a late particle gains |q|V sin(w dt), above transition it loses it.
      for (zp, sgn) in ((PhaseRef.BelowTransition, +1), (PhaseRef.AboveTransition, -1))
        exp_dE = sgn * 1000 * sin(w * dt)
        dS = gain(s, SaganCavity(n_cells = 0, L_active = 0.0), zp, z)
        dO = gain(s, SciBmadStandard(), zp, z)
        println(rpad(nameof(s), 9), rpad(string(zp), 32), " Sagan dE = ", dS, "  ordinary dE = ", dO,
                "  expected = ", exp_dE)
        global bad |= abs(dS + exp_dE) < 1e-4 * abs(exp_dE) && abs(dO - exp_dE) < 1e-4 * abs(exp_dE)
      end
    end
    # --- Independent recheck: multi-turn synchrotron stability in a drift-only ring -------------
    # A drift-only ring has momentum compaction 0, i.e. it is below transition (slip -1/gamma^2).
    # A correct BelowTransition cavity is stable there; a correct AboveTransition one is unstable.
    function ring_growth(method, zp)
      M = massof(Species("proton")); pc = M; s0 = Species("proton")
      cav = RFCavity(L = (method isa SaganCavity ? 0.0 : 1e-6), voltage = 4.5e7, rf_frequency = 1e7,
                     zero_phase = zp, tracking_method = method)
      br = Branch([Beamline([Marker()], species_ref = s0, pc_ref = pc), Beamline([cav, Drift(L = 10.0)], dE_ref = 0.0)])
      b = Bunch([0.0 0.0 0.0 0.0 1e-3 0.0], species = s0, p_over_q_ref = pc / C_LIGHT)
      track!(b, br.beamlines[1]; use_explicit_SIMD = false)
      zmax = 0.0
      for _ in 1:40
        track!(b, br.beamlines[2]; use_explicit_SIMD = false); zmax = max(zmax, abs(b.v[1, 5]))
      end
      return zmax / 1e-3
    end
    for (lab, m) in (("ordinary", SciBmadStandard()), ("Sagan", SaganCavity(n_cells = 0, L_active = 0.0))),
        zp in (PhaseRef.BelowTransition, PhaseRef.AboveTransition)
      println("recheck: drift ring, ", rpad(lab, 9), rpad(string(zp), 28), " max|z|/z0 over 40 turns = ", ring_growth(m, zp))
    end
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | BeamTracking 0.8.7 (`d8e45fdc25db`), source (SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, Float64, one thread |

