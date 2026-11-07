---
title: "COL-05: DAMap keyword constructor cannot infer nv without v"
hide:
  - toc
---

# COL-05 — `DAMap` keyword constructor cannot infer `nv` without `v`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on NonlinearNormalForm release [v0.5.3](https://github.com/bmad-sim/NonlinearNormalForm.jl/tree/v0.5.3) (`7e40539`) |
| Source | [`NonlinearNormalForm.jl/src/map.jl:319-341`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L319-L341)<br>[`map.jl:288`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L288)<br>[`map.jl:319-336`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L319-L336)<br>[`map.jl:337-341`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L337-L341) |

Constructing a `DAMap` with keywords fails when neither `nv`, the number of orbital variables, nor the orbital rays `v` are given. The constructor promises to infer `nv` from the other supplied data or from its default, but afterwards overwrites that inference with `length(v)`, which fails when `v` is `nothing`. With a six-variable, first-order GTPSA (truncated power series) descriptor, passing a 6×6 identity matrix as `v_matrix`, a six-entry reference vector `v0`, or no orbital data raises `MethodError` instead of constructing a map, while the same calls with `nv=6` succeed. Calls that supply `v` also work, and SciBmad's Twiss constructor avoids the failure by passing `nv` explicitly.

Run `julia COL-05.jl`.

??? example "Reproducer: COL-05.jl"

    [Download COL-05.jl](COL-05.jl)

    ```julia
    # COL-05: `DAMap` keyword constructor cannot infer `nv` without `v`
    #
    # Constructing a `DAMap` with keywords fails when neither `nv`, the number of orbital variables, nor
    # the orbital rays `v` are given. The constructor promises to infer `nv` from the other supplied
    # data or from its default, but afterwards overwrites that inference with `length(v)`, which fails
    # when `v` is `nothing`. With a six-variable, first-order GTPSA (truncated power series) descriptor,
    # passing a 6×6 identity matrix as `v_matrix`, a six-entry reference vector `v0`, or no orbital data
    # raises `MethodError` instead of constructing a map, while the same calls with `nv=6` succeed.
    # Calls that supply `v` also work, and SciBmad's Twiss constructor avoids the failure by passing
    # `nv` explicitly.
    #
    # The script sets up a six-variable, first-order GTPSA descriptor and constructs a `DAMap` with
    # `v_matrix = I(6)`, with `v0 = zeros(6)` and with no orbital argument, each time without `nv` and,
    # for comparison, with `nv=6`. As a control, it constructs a map from six orbital rays `v`, the one
    # branch that works.
    using SciBmad, LinearAlgebra
    const NNF = SciBmad.NNF
    const TI = SciBmad.TI

    init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(dynamic_descriptor = Descriptor(6, 1))
    # The constructor has nv::Union{Integer,Nothing}=nothing and tries to infer nv from v0, v, v_matrix,
    # s or DEFAULT_NVARS, then overwrites the inferred value with length(v) (or length(v)-1), raising
    # MethodError (length(nothing)) when v is omitted. `v0=zeros(6)` must infer six variables. Whether a
    # matrix-only `v_matrix=I(6)` call infers six variables or five variables and one parameter is an
    # inference rule upstream has not settled; the crash establishes neither count.
    cases = [("v_matrix = I(6) matrix", (; v_matrix = Matrix(1.0I, 6, 6))),
             ("v0 = zeros(6)",          (; v0 = zeros(6))),
             ("no orbital argument (DEFAULT_NVARS)", (;))]
    nfail = 0
    for (label, kw) in cases
        ref = DAMap(; init = init, nv = 6, kw...)
        try
            m = DAMap(; init = init, kw...)
            println(label, ": nv inferred = ", NNF.nvars(m), "  (explicit nv=6 gives ", NNF.nvars(ref), ")")
        catch err
            global nfail += 1
            st = stacktrace(catch_backtrace())
            k = findfirst(f -> occursin("map.jl", string(f.file)), st)
            println(label, ": ", first(split(sprint(showerror, err), "\n")), "  at ",
                    isnothing(k) ? "?" : "$(basename(string(st[k].file))):$(st[k].line)",
                    "  (explicit nv=6 succeeds, nvars = ", NNF.nvars(ref), ")")
        end
    end
    # Independent recheck: the `v` branch, the only one that defines v, succeeds.
    mv = DAMap(v = DAMap(init = init, nv = 6).v)
    println("recheck: DAMap(v = <6 TPS>) -> nvars = ", NNF.nvars(mv))
    println(nfail > 0 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source, Julia 1.13.2, CPU, Float64, serial |

