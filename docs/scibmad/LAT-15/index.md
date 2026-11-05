---
title: "LAT-15: GPU batch documentation assigns a raw array instead of a BatchParam"
hide:
  - toc
---

# LAT-15 — GPU batch documentation assigns a raw array instead of a `BatchParam`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`SciBmad.jl/docs/src/batch.md:105`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/batch.md#L105)<br>[`BeamTracking.jl/src/batch.jl:303-312`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/batch.jl#L303-L312)<br>[`SciBmad.jl/docs/src/batch.md:48`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/batch.md#L48)<br>[`:74`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/batch.md#L74) |

The GPU example of the batch documentation assigns a raw `CuArray` of quadrupole strengths to `fodo.context.K1`, but only a `BatchParam` is expanded into per-particle values, and any other value passes through unchanged. The CPU examples on the same page wrap the array in `BatchParam`, and a source search finds no GPU-array conversion in Beamlines or BeamTracking. Run on the CPU with an ordinary array in place of the `CuArray`, the page's FODO cell fails with `MethodError: no method matching ^(::Vector{Float64}, ::Int64)`, while the same strengths wrapped in `BatchParam` track.

??? example "Reproducer: LAT-15.jl"

    [Download LAT-15.jl](LAT-15.jl)

    ```julia
    # LAT-15: GPU batch documentation assigns a raw array instead of a `BatchParam`
    #
    # The GPU example of the batch documentation assigns a raw `CuArray` of quadrupole strengths to
    # `fodo.context.K1`, but only a `BatchParam` is expanded into per-particle values, and any other
    # value passes through unchanged. The CPU examples on the same page wrap the array in `BatchParam`,
    # and a source search finds no GPU-array conversion in Beamlines or BeamTracking. Run on the CPU
    # with an ordinary array in place of the `CuArray`, the page's FODO cell fails with `MethodError: no
    # method matching ^(::Vector{Float64}, ::Int64)`, while the same strengths wrapped in `BatchParam`
    # track.
    #
    # The script builds the page's FODO cell, with quadrupole strengths `Kn1` = ±K1 and ten values of K1
    # from 0.1 to 0.55, and tracks it once with `BatchParam(K)` and once with the plain array K, a CPU
    # stand-in for the `CuArray` that takes the same non-`BatchParam` path. The plain-array assignment
    # should behave as the `BatchParam` scan. No GPU is used.
    using SciBmad
    @elements begin
      qf = Quadrupole(Kn1=DefExpr(c -> c.K1), L=0.4)
      d = Drift(L=0.5)
      qd = Quadrupole(Kn1=DefExpr(c -> -c.K1), L=0.4)
    end
    fodo = Beamline([qf, d, qd, d], E_ref=18e9, species_ref=Species("electron"))
    K = collect(range(start=0.1, stop=0.55, length=10))
    v0 = repeat([1e-3 0 0 0 0 1e-3], 10, 1)
    fodo.context.K1 = BatchParam(K)
    xb = track(fodo; v0=v0).v[:, 1, end]
    println("BatchParam(K):  x = ", xb)
    # Only a BatchParam is lowered to per-particle values; any other value passes through unchanged, and no
    # GPU-array conversion exists in Beamlines or BeamTracking, so a CPU array takes the same path as a CuArray.
    fodo.context.K1 = K                     # the documented GPU pattern, with a CPU array
    r = try
        track(fodo; v0=v0).v[:, 1, end]
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("raw array K:    x = ", r)
    println("expected (doc): raw-array assignment behaves as the BatchParam scan above")
    println(r == xb ? "NOT REPRODUCED" : "DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |

