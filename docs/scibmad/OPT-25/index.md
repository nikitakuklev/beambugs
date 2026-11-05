---
title: "OPT-25: VectorField keyword constructor fails unless v is given"
hide:
  - toc
---

# OPT-25 — `VectorField` keyword constructor fails unless `v` is given

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on NonlinearNormalForm main (`7e40539`) |
| Source | [`src/vectorfield.jl:138-180`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L138-L180)<br>[`nv = length(v)`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L156)<br>[`setray!`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L175)<br>[`vectorfield.jl:138-145`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L138-L145)<br>[`:162`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/vectorfield.jl#L162) |

With a valid TPSA definition `init`, NonlinearNormalForm's `VectorField` keyword constructor throws `MethodError: no method matching length(::Nothing)` whenever the orbital rays `v` are omitted. This blocks `init` with a matrix `v_matrix`, `init` with spin-only input, and `init` alone, because the constructor counts `v` before reaching the branch intended to allocate missing rays. For a 2×2 rotation generator on `Descriptor(2, 2)`, explicit rays build the expected field and the `DAMap` matrix constructor reproduces the matrix exactly. Literal `VectorField()` instead fails earlier because no TPSA definition was supplied; it does not demonstrate this counting error.

??? example "Reproducer: OPT-25.jl"

    [Download OPT-25.jl](OPT-25.jl)

    ```julia
    # OPT-25: `VectorField` keyword constructor fails unless `v` is given
    #
    # With a valid TPSA definition `init`, NonlinearNormalForm's `VectorField` keyword constructor
    # throws `MethodError: no method matching length(::Nothing)` whenever the orbital rays `v` are
    # omitted. This blocks `init` with a matrix `v_matrix`, `init` with spin-only input, and `init`
    # alone, because the constructor counts `v` before reaching the branch intended to allocate missing
    # rays. For a 2×2 rotation generator on `Descriptor(2, 2)`, explicit rays build the expected field
    # and the `DAMap` matrix constructor reproduces the matrix exactly. Literal `VectorField()` instead
    # fails earlier because no TPSA definition was supplied; it does not demonstrate this counting
    # error.
    #
    # The script builds a 2×2 rotation generator M on `Descriptor(2, 2)` and calls the keyword
    # constructor with `init` and `v_matrix=M`, with `init` and a constant quaternion `q` only, and with
    # `init` alone. As controls, it builds the same field from the explicit rays M·(x, p) and passes
    # `v_matrix=M` to the `DAMap` keyword constructor with explicit `nv=2`, which uses the same ray
    # setter, and prints the linear coefficients of both.
    using SciBmad
    const NNF = NonlinearNormalForm
    const TI = SciBmad.TI
    d = Descriptor(2, 2); x, p = vars(d)
    init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(dynamic_descriptor=d)
    M = [0.0 1.0; -1.0 0.0]
    res(f) = try f(); "ok" catch err; first(split(sprint(showerror, err), '\n')) end
    e = res(() -> VectorField(init=init, v_matrix=M))
    println("VectorField(init, v_matrix=M)       -> ", e)
    println("VectorField(init, q=[1,0,0,0])      -> ", res(() -> VectorField(init=init, q=[1.0, 0, 0, 0])))
    println("VectorField(init)                   -> ", res(() -> VectorField(init=init)))
    F = VectorField(v=[M[1, 1] * x + M[1, 2] * p, M[2, 1] * x + M[2, 2] * p])
    println("independent: VectorField(v = M*[x,p]) rays = ", [TI.geti(F.v[i], j) for i in 1:2, j in 1:2])
    m = DAMap(init=init, nv=2, v_matrix=M)
    println("independent: DAMap(init, nv=2, v_matrix=M) rays = ", [TI.geti(m.v[i], j) for i in 1:2, j in 1:2])
    println(occursin("length(::Nothing)", e) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-24 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source with the proposed fix for COL-05 applied, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-24 | NonlinearNormalForm 0.5.3 (`7e4053985736`), pinned and main-branch sources, Julia 1.13.2, CPU, Float64 |

