---
title: "OPT-14: Batched Newton returns the previous iterate's f and jac"
hide:
  - toc
---

# OPT-14 — Batched Newton returns the previous iterate's `f` and `jac`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on BatchSolve main (`ac41c09`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`src/rootfinders/newton.jl:254-278`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/rootfinders/newton.jl#L254-L278)<br>[`:266`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/rootfinders/newton.jl#L266)<br>[`:274`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/rootfinders/newton.jl#L274)<br>[`:276`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/rootfinders/newton.jl#L276)<br>[`newton.jl:172-179`](https://github.com/mattsignorelli/BatchSolve.jl/blob/ac41c095255df9a3d4f33c4242a4e5d69e32a7ec/src/rootfinders/newton.jl#L172-L179)<br>[`SciBmad.jl/src/closed_orbit.jl:135`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/closed_orbit.jl#L135)<br>[`one-torus.jl:64-69`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/experimental/one-torus.jl#L64-L69) |

BatchSolve's batched `newton` returns stale `f` and `jac` for lanes (problems in the batch) that the relative-step test accepts in the final iteration: it updates the iterate, then exits before recomputing the promised final residual and Jacobian. For x² − 2 from x = 1 with `reltol = 0.4`, the accepted iterate is 1.5, but the returned residual and Jacobian are −1.000 and 2.000 instead of 0.2500 and 3.000; the non-batched solver returns the latter values. The accepted iterate is unchanged by this diagnostic defect, and residual-test exits are unaffected. By source inspection, batched closed-orbit finding uses this path and experimental aperture code reads the Jacobian as I − M, with M the one-turn matrix.

??? example "Reproducer: OPT-14.jl"

    [Download OPT-14.jl](OPT-14.jl)

    ```julia
    # OPT-14: Batched Newton returns the previous iterate's `f` and `jac`
    #
    # BatchSolve's batched `newton` returns stale `f` and `jac` for lanes (problems in the batch) that
    # the relative-step test accepts in the final iteration: it updates the iterate, then exits before
    # recomputing the promised final residual and Jacobian. For x² − 2 from x = 1 with `reltol = 0.4`,
    # the accepted iterate is 1.5, but the returned residual and Jacobian are −1.000 and 2.000 instead
    # of 0.2500 and 3.000; the non-batched solver returns the latter values. The accepted iterate is
    # unchanged by this diagnostic defect, and residual-test exits are unaffected. By source inspection,
    # batched closed-orbit finding uses this path and experimental aperture code reads the Jacobian as I
    # − M, with M the one-turn matrix.
    #
    # The script solves f(x) = x² − 2 from x = 1 with `reltol = 0.4`, once with the batched solver (two
    # lanes) and once with the non-batched solver as the control. One step gives x = 1.5, and |Δx| = 0.5
    # < 0.4·1.5 ends the iteration, so the returned `f` and `jac` must equal f(1.5) = 0.25 and f′(1.5) =
    # 3.
    using SciBmad
    const BS = SciBmad.BatchSolve
    f(x) = x .^ 2 .- 2
    # In the batched branch x is updated, the relative-step test marks the lane converged, and the loop
    # breaks before the residual and Jacobian are evaluated again; the non-batched branch re-evaluates
    # before its step test.
    b = BS.newton(f, fill(1.0, 2, 1); batchdim = 1, reltol = 0.4, abstol = 1e-14)
    n = BS.newton(f, [1.0]; reltol = 0.4, abstol = 1e-14)
    println("batched    : u=", vec(b.u), " f=", vec(b.f), " jac=", Matrix(b.jac), " retcode=", vec(b.retcode), " iters=", vec(b.iters))
    println("expected   : f(u)=", vec(f(b.u)), " jac diag=", 2 .* vec(b.u))
    println("non-batched: u=", n.u, " f=", n.f, " jac=", Matrix(n.jac), " retcode=", n.retcode, " iters=", n.iters)
    println(maximum(abs, b.f .- f(b.u)) > 1e-12 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BatchSolve 0.1.5 (`ac41c095255d`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BatchSolve 0.1.5 (`ac41c095255d`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BatchSolve 0.1.5 (`ac41c095255d`), release tag clones, Julia 1.13.2 guide wrapper, CPU, Float64 |

