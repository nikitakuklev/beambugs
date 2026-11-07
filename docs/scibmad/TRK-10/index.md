---
title: "TRK-10: Methods other than RungeKutta silently ignore EMFieldParams"
hide:
  - toc
---

# TRK-10 — Methods other than `RungeKutta` silently ignore `EMFieldParams`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [ext/BeamTrackingBeamlinesExt/unpack_bl.jl:139–318](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L139-L318)<br>[ext/BeamTrackingBeamlinesExt/unpack_bl.jl:357–366](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L357-L366)<br>[ext/BeamTrackingBeamlinesExt/rungekutta.jl:16–21](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/rungekutta.jl#L16-L21)<br>[ext/BeamTrackingBeamlinesExt/rungekutta.jl:70](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/rungekutta.jl#L70)<br>[src/misc.jl:147–169](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/misc.jl#L147-L169)<br>[src/element.jl:12](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/element.jl#L12)<br>[ext/BeamTrackingBeamlinesExt/unpack_bl.jl:263](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L263)<br>[ext/BeamTrackingBeamlinesExt/rungekutta.jl:39–51](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/rungekutta.jl#L39-L51) |

Adding `EMFieldParams` to an element while keeping the default tracking method silently omits the supplied electromagnetic field. The field-definition contract and an independent Lorentz-impulse calculation require its force, which `RungeKutta` applies correctly. A proton with reference momentum p₀c = 2mc², twice its rest energy, crossing 1.000 m of uniform vertical field 0.01000 T exits with normalized horizontal momentum 0 instead of −1.598 × 10⁻³. In tests `SciBmadStandard`, `Symplectic`, `DriftKick` and `Exact` omit magnetic fields and `SaganCavity` omits a superposed electric field. Source inspection also finds unchecked `EMultipoleParams` on the map, patch and four-potential branches.

Run `julia TRK-10.jl`.

??? example "Reproducer: TRK-10.jl"

    [Download TRK-10.jl](TRK-10.jl)

    ```julia
    # TRK-10: Methods other than `RungeKutta` silently ignore `EMFieldParams`
    #
    # Adding `EMFieldParams` to an element while keeping the default tracking method silently omits the
    # supplied electromagnetic field. The field-definition contract and an independent Lorentz-impulse
    # calculation require its force, which `RungeKutta` applies correctly. A proton with reference
    # momentum p₀c = 2mc², twice its rest energy, crossing 1.000 m of uniform vertical field 0.01000 T
    # exits with normalized horizontal momentum 0 instead of −1.598 × 10⁻³. In tests `SciBmadStandard`,
    # `Symplectic`, `DriftKick` and `Exact` omit magnetic fields and `SaganCavity` omits a superposed
    # electric field. Source inspection also finds unchecked `EMultipoleParams` on the map, patch and
    # four-potential branches.
    #
    # The script tracks a proton with p₀c = 2mc² through a 1 m `LineElement` whose `em_field` is a
    # uniform vertical field of 0.01 T, with `SciBmadStandard` and with `RungeKutta(n_steps=50)`. It
    # compares the exit horizontal momentum with the impulse −B_y·L/R, where R is the magnetic rigidity,
    # and would also accept an error; `RungeKutta` is the control that applies the field.
    using SciBmad

    const sp = Species("proton"); const pc = 2 * massof(sp); const R = pc / C_LIGHT
    field(x, y, s, t, p = nothing) = (0.0, 0.0, 0.0, 0.0, 0.01, 0.0)   # uniform By = 0.01 T
    function out(tm)
      ele = LineElement(L = 1.0, em_field = field, tracking_method = tm)
      b = Bunch(zeros(1, 6), species = sp, p_over_q_ref = R)
      try
        track!(b, Beamline([ele], species_ref = sp, pc_ref = pc); use_explicit_SIMD = false)
        return vec(b.v)
      catch err
        return sprint(showerror, err)
      end
    end
    # EMFieldParams defines the element's field and is an active group; other unsupported active groups
    # are rejected with an error, so the field must deflect the beam by px = -By L/R or raise an error.
    o_std = out(SciBmadStandard()); o_rk = out(RungeKutta(n_steps = 50))
    println("SciBmadStandard: ", o_std)
    println("RungeKutta:      ", o_rk)
    println("expected px ~ -By L / R = ", -0.01 / R, " (or an error)")
    bad = o_std isa Vector && all(iszero, o_std) && o_rk isa Vector && abs(o_rk[2] + 0.01 / R) < 1e-6
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | BeamTracking 0.8.7 (`d8e45fdc25db`), source (SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, Float64, one thread |

