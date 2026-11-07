---
title: "OPT-15: Allocating newton/brent document caller mutation they do not do"
hide:
  - toc
---

# OPT-15 — Allocating `newton`/`brent` document caller mutation they do not do

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on BatchSolve main (`ac41c09`, head build) |
| Source | [`src/rootfinders/newton.jl:30`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/rootfinders/newton.jl#L30)<br>[`:59`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/rootfinders/newton.jl#L59)<br>[`src/minimizers/brent.jl:24`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/minimizers/brent.jl#L24)<br>[`:38`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/minimizers/brent.jl#L38) |

The docstrings of BatchSolve's allocating `newton` and `brent` say that the returned solution `u` is the caller's `x`, mutated in place, but both work on a copy and leave `x` unchanged. The in-place `brent!` does alias its argument, so the text describes the in-place functions. Calling `newton` from x = [2.0] returns u = [1.000] and leaves x = [2.0], with `u === x` false. Code that follows the docstring and reads `x` after the call gets the initial guess. The allocating behavior itself is sensible; only the documentation is wrong.

Run `julia OPT-15.jl`.

??? example "Reproducer: OPT-15.jl"

    [Download OPT-15.jl](OPT-15.jl)

    ```julia
    # OPT-15: Allocating `newton`/`brent` document caller mutation they do not do
    #
    # The docstrings of BatchSolve's allocating `newton` and `brent` say that the returned solution `u`
    # is the caller's `x`, mutated in place, but both work on a copy and leave `x` unchanged. The
    # in-place `brent!` does alias its argument, so the text describes the in-place functions. Calling
    # `newton` from x = [2.0] returns u = [1.000] and leaves x = [2.0], with `u === x` false. Code that
    # follows the docstring and reads `x` after the call gets the initial guess. The allocating behavior
    # itself is sensible; only the documentation is wrong.
    #
    # The script calls `newton` on v² − 1 from x = [2.0] and `brent` on (v − 0.25)² from x = [0.0], and
    # checks whether the returned `u` is the caller's `x` and whether `x` changed. As a control, the
    # in-place `brent!` on the same minimization returns its argument, which then holds the solution.
    using SciBmad
    const BS = SciBmad.BatchSolve
    x = [2.0]
    # Both allocating wrappers pass copy(x) to the in-place solver.
    s = BS.newton(v -> v .^ 2 .- 1, x)
    println("newton: u=", s.u, "  x after call=", x, "  u === x: ", s.u === x)
    xb = [0.0]
    t = BS.brent(v -> (v .- 0.25) .^ 2, xb; xa = [-1.0], xb = [1.0])
    println("brent : u=", t.u, "  x after call=", xb, "  u === x: ", t.u === xb)
    xi = [0.0]; yi = similar(xi)
    ti = BS.brent!((y, v) -> (y .= (v .- 0.25) .^ 2; y), yi, xi; xa = [-1.0], xb = [1.0])
    println("independent: brent! u === x: ", ti.u === xi, "  x after call=", xi)
    println(!(s.u === x) && !(t.u === xb) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | BatchSolve 0.1.5 (`ac41c095255d`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | BatchSolve 0.1.5 (`ac41c095255d`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BatchSolve 0.1.5 (`ac41c095255d`), release sources, Julia 1.13.2, CPU, Float64 |

