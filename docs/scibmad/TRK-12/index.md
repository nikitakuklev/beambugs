---
title: "TRK-12: Sagan tracking guide uses nonexistent names"
hide:
  - toc
---

# TRK-12 — Sagan tracking guide uses nonexistent names

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Source | [docs/src/sagancavity-tracking.md:16](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-tracking.md#L16)<br>[docs/src/sagancavity-tracking.md:28](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-tracking.md#L28)<br>[docs/src/sagancavity-tracking.md:32](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-tracking.md#L32)<br>[docs/src/sagancavity-tracking.md:42](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-tracking.md#L42)<br>[docs/src/sagancavity-tracking.md:56–58](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-tracking.md#L56-L58)<br>[docs/src/sagancavity-physics.md:37](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-physics.md#L37)<br>[docs/src/sagancavity-physics.md:95](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-physics.md#L95)<br>[src/tracking_methods.jl:79–89](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/tracking_methods.jl#L79-L89)<br>[src/rf.jl:9](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/rf.jl#L9)<br>[src/rf.jl:23](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/rf.jl#L23)<br>[src/rf.jl:61–68](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/rf.jl#L61-L68) |

The `SaganCavity` tracking guide's examples fail when copied, because they use the keyword `num_cells`, the nonexistent enum name `PhaseReference`, and `zero_phase` as a `SaganCavity` keyword. The constructor and export list instead provide `n_cells` and `PhaseRef`, and `zero_phase` belongs to the cavity element's `RFParams`. The guide's two-cell call with active length 1.600 m raises `MethodError`, while `SaganCavity(n_cells=2, L_active=1.6)` succeeds; passing `zero_phase` to that constructor also raises `MethodError`. The physics guide repeats the wrong cell-count keyword, so users following either guide are affected.

??? example "Reproducer: TRK-12.jl"

    [Download TRK-12.jl](TRK-12.jl)

    ```julia
    # TRK-12: Sagan tracking guide uses nonexistent names
    #
    # The `SaganCavity` tracking guide's examples fail when copied, because they use the keyword
    # `num_cells`, the nonexistent enum name `PhaseReference`, and `zero_phase` as a `SaganCavity`
    # keyword. The constructor and export list instead provide `n_cells` and `PhaseRef`, and
    # `zero_phase` belongs to the cavity element's `RFParams`. The guide's two-cell call with active
    # length 1.600 m raises `MethodError`, while `SaganCavity(n_cells=2, L_active=1.6)` succeeds;
    # passing `zero_phase` to that constructor also raises `MethodError`. The physics guide repeats the
    # wrong cell-count keyword, so users following either guide are affected.
    #
    # The script calls the guide's `SaganCavity(num_cells=2, L_active=1.6)` and
    # `SaganCavity(zero_phase=PhaseRef.BelowTransition)` and checks whether `PhaseReference` is defined
    # in SciBmad. The control `SaganCavity(n_cells=2, L_active=1.6)` must succeed.
    using SciBmad

    function attempt(label, f)
      try
        f(); println(rpad(label, 46), " OK"); return false
      catch err
        println(rpad(label, 46), " ", typeof(err), ": ", first(split(sprint(showerror, err), '\n'))); return true
      end
    end
    # The constructor keywords are n_cells, L_active, radiation_damping_on and
    # radiation_fluctuations_on; the enum is PhaseRef, and zero_phase is a field of RFParams.
    e1 = attempt("SaganCavity(num_cells = 2, L_active = 1.6)", () -> SaganCavity(num_cells = 2, L_active = 1.6))
    e2 = attempt("SaganCavity(zero_phase = PhaseRef.BelowTransition)", () -> SaganCavity(zero_phase = PhaseRef.BelowTransition))
    e3 = !isdefined(SciBmad, :PhaseReference)
    println(rpad("isdefined(SciBmad, :PhaseReference)", 46), " ", !e3)
    ok = attempt("control: SaganCavity(n_cells = 2, L_active = 1.6)", () -> SaganCavity(n_cells = 2, L_active = 1.6))
    println(e1 && e2 && e3 && !ok ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |

