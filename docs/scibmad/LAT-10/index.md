---
title: "LAT-10: Exported ParticleView cannot be constructed"
hide:
  - toc
---

# LAT-10 — Exported `ParticleView` cannot be constructed

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`BeamTracking.jl/src/types.jl:132-148`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/types.jl#L132-L148)<br>[`BeamTracking.jl/src/BeamTracking.jl:26`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/BeamTracking.jl#L26) |

Constructing the exported `ParticleView(bunch, i)`, meant as a view of particle i of a `Bunch`, fails with “too many parameters for type ParticleView”: the struct declares six type parameters, but its inner constructor supplies eight, one per field, as reflection confirms. For a bunch whose particle 2 has coordinates [2, 5, 8, 11, 14, 17], the call raises instead of returning that row. Source reading shows a second fault behind the first: the outer constructor takes coordinate columns `view(v, :, i)` and `view(q, :, i)`, whereas particles are rows of the N×6 and N×4 arrays, so the view would hold wrong data once construction works. No code in the SciBmad packages uses `ParticleView`.

Run `julia LAT-10.jl`.

??? example "Reproducer: LAT-10.jl"

    [Download LAT-10.jl](LAT-10.jl)

    ```julia
    # LAT-10: Exported `ParticleView` cannot be constructed
    #
    # Constructing the exported `ParticleView(bunch, i)`, meant as a view of particle i of a `Bunch`,
    # fails with “too many parameters for type ParticleView”: the struct declares six type parameters,
    # but its inner constructor supplies eight, one per field, as reflection confirms. For a bunch whose
    # particle 2 has coordinates [2, 5, 8, 11, 14, 17], the call raises instead of returning that row.
    # Source reading shows a second fault behind the first: the outer constructor takes coordinate
    # columns `view(v, :, i)` and `view(q, :, i)`, whereas particles are rows of the N×6 and N×4 arrays,
    # so the view would hold wrong data once construction works. No code in the SciBmad packages uses
    # `ParticleView`.
    #
    # The script builds a three-particle `Bunch` whose coordinates are 1 to 18 and calls
    # `ParticleView(bunch, 2)`, expecting row 2 of the coordinate matrix. It also prints the numbers of
    # type parameters and fields of `ParticleView`.
    using SciBmad
    const BT = SciBmad.BeamTracking
    b = Bunch(v=reshape(collect(1.0:18.0), 3, 6))
    # ParticleView declares six type parameters {B,T,S,V,Q,W}, but its inner constructor passes the types of all
    # eight fields (index::Int and species::Species included).
    r = try
        pv = ParticleView(b, 2)
        pv.v
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("observed ParticleView(bunch, 2).v = ", r)
    println("expected particle row 2 of the N x 6 matrix = ", b.v[2, :])
    T = Base.unwrap_unionall(ParticleView)
    println("type parameters = ", length(T.parameters), ", fields = ", fieldcount(T))
    println(r isa String ? "DEFECT REPRODUCED" : (r == b.v[2, :] ? "NOT REPRODUCED" : "DEFECT REPRODUCED"))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

