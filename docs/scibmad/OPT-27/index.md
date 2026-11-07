---
title: "OPT-27: q_map initialization is skipped unless q is supplied"
hide:
  - toc
---

# OPT-27 — `q_map` initialization is skipped unless `q` is supplied

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on NonlinearNormalForm release [v0.5.3](https://github.com/bmad-sim/NonlinearNormalForm.jl/tree/v0.5.3) (`7e40539`) |
| Source | [`src/vectorfield.jl:176-177`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L176-L177)<br>[`src/map.jl:387-388`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L387-L388)<br>[`vectorfield.jl:164-169`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L164-L169)<br>[`src/set.jl:56-84`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/set.jl#L56-L84) |

NonlinearNormalForm's `VectorField`, `DAMap` and `TPSAMap` keyword constructors allocate spin storage when given `q_map`, a matrix of linear quaternion coefficients, but leave those coefficients zero unless the constant quaternion `q` is also given. With explicit orbital rays and map dimensions, requesting (q1[x], q3[p]) = (0.375, −0.125) returns (0, 0) in all three constructors. An explicit zero `q` or a direct `setquat!` call preserves both coefficients exactly. A pure linear `VectorField` spin generator legitimately has a zero constant quaternion, so this is a genuine spin-initialization failure. The map tests establish the omitted coefficients without asserting that a zero constant quaternion represents a physical spin rotation.

Run `julia OPT-27.jl`.

??? example "Reproducer: OPT-27.jl"

    [Download OPT-27.jl](OPT-27.jl)

    ```julia
    # OPT-27: `q_map` initialization is skipped unless `q` is supplied
    #
    # NonlinearNormalForm's `VectorField`, `DAMap` and `TPSAMap` keyword constructors allocate spin
    # storage when given `q_map`, a matrix of linear quaternion coefficients, but leave those
    # coefficients zero unless the constant quaternion `q` is also given. With explicit orbital rays and
    # map dimensions, requesting (q1[x], q3[p]) = (0.375, −0.125) returns (0, 0) in all three
    # constructors. An explicit zero `q` or a direct `setquat!` call preserves both coefficients
    # exactly. A pure linear `VectorField` spin generator legitimately has a zero constant quaternion,
    # so this is a genuine spin-initialization failure. The map tests establish the omitted coefficients
    # without asserting that a zero constant quaternion represents a physical spin rotation.
    #
    # The script calls each constructor with a `q_map` requesting q1[x] = 0.375 and q3[p] = −0.125,
    # first alone and then with an explicit zero `q`, then applies `setquat!(q_map=…)` directly to the
    # first result, and reads the two coefficients each time. It supplies orbital rays, and explicit
    # `nv=2` for the maps, so that the separate dimension-inference faults stay out of the way.
    using SciBmad, LinearAlgebra
    const NNF = NonlinearNormalForm
    BLAS.set_num_threads(1)
    x,p = vars(Descriptor(2,3))
    checks = Bool[]
    qm=zeros(4,2); qm[2,1]=0.375; qm[4,2]=-0.125
    for kind in (VectorField, DAMap, NNF.TPSAMap)
        if kind === VectorField
            a = kind(v=[zero(x),zero(p)], q_map=qm)
            c = kind(v=[zero(x),zero(p)], q=zeros(4), q_map=qm)
        else
            a = kind(nv=2,v=[x,p], q_map=qm)
            c = kind(nv=2,v=[x,p], q=zeros(4), q_map=qm)
        end
        println(kind," alone = ",(a.q.q1[[1,0]], a.q.q3[[0,1]])," explicit q = ",(c.q.q1[[1,0]], c.q.q3[[0,1]]))
        @assert a.q !== nothing
        push!(checks, a.q.q1[[1,0]] == 0 && a.q.q3[[0,1]] == 0)
        @assert c.q.q1[[1,0]] == qm[2,1] && c.q.q3[[0,1]] == qm[4,2]
        NNF.setquat!(a.q; q_map=qm)
        println("independent: direct setquat! = ",(a.q.q1[[1,0]], a.q.q3[[0,1]]))
        @assert a.q.q1[[1,0]] == qm[2,1] && a.q.q3[[0,1]] == qm[4,2]
    end
    println("VectorField zero constant is a valid spin generator")

    println(all(checks) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source, Julia 1.13.2, CPU, Float64, serial |

