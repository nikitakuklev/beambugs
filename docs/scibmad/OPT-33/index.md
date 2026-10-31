---
title: "OPT-33: Scaled identity addition and subtraction ignore the scale"
hide:
  - toc
---

# OPT-33 — Scaled identity addition and subtraction ignore the scale

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on NonlinearNormalForm main (`7e40539`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`src/operators.jl:87`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/operators.jl#L87)<br>[`:126`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/operators.jl#L126)<br>[`:95`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/operators.jl#L95)<br>[`:134`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/operators.jl#L134)<br>[`docs/src/index.md:21`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/docs/src/index.md#L21) |

For a NonlinearNormalForm `DAMap` M and a scaled identity λI, `M + λI`, `λI + M` and `M - λI` ignore λ and change each orbital diagonal coefficient by +1 or −1. Coefficient arithmetic requires M + 0I = M and a diagonal change of λ, which explicitly constructed scaled-identity maps confirm. For a map with diagonal (2, 3), `M + 0I` returns (3, 4), while `M - 3I` returns (1, 2) instead of (−1, 0). Unit-scale addition and right subtraction avoid this scale error. Source inspection establishes the analogous `TPSAMap`, `VectorField` and quaternion-scalar paths; downstream optics effects were not measured.

??? example "Reproducer: OPT-33.jl"

    [Download OPT-33.jl](OPT-33.jl)

    ```julia
    # OPT-33: Scaled identity addition and subtraction ignore the scale
    #
    # For a NonlinearNormalForm `DAMap` M and a scaled identity λI, `M + λI`, `λI + M` and `M - λI`
    # ignore λ and change each orbital diagonal coefficient by +1 or −1. Coefficient arithmetic requires
    # M + 0I = M and a diagonal change of λ, which explicitly constructed scaled-identity maps confirm.
    # For a map with diagonal (2, 3), `M + 0I` returns (3, 4), while `M - 3I` returns (1, 2) instead of
    # (−1, 0). Unit-scale addition and right subtraction avoid this scale error. Source inspection
    # establishes the analogous `TPSAMap`, `VectorField` and quaternion-scalar paths; downstream optics
    # effects were not measured.
    #
    # The script builds the two-component `DAMap` (2x + 0.125p + 0.75x², −0.2x + 3p − 0.3x²) on
    # `Descriptor(2, 3)` and, for λ = 0, 2, 3 and −3, compares `M + λI`, `λI + M` and `M - λI`
    # coefficient by coefficient with ordinary arithmetic against the map (λx, λp); addition at λ = 1 is
    # a passing control. For the unit-diagonal map (x, p + 0.25x²) it shows that `M - 3I` and `M +
    # (-3I)` give two different wrong diagonals, although both must be (−2, −2).
    using SciBmad, LinearAlgebra, Printf
    const NNF = SciBmad.NonlinearNormalForm
    BLAS.set_num_threads(1)
    x,p = vars(Descriptor(2,3))
    m = DAMap(nv=2, v=[2x+.125p+.75x*x, -.2x+3p-.3x*x])
    for scale in (0., 2., 3., -3.)
        J = scale*I
        scaled_identity = DAMap(nv=2, v=[scale*x, scale*p])
        for (label, actual, oracle) in (("M+lambdaI",m+J,m+scaled_identity), ("lambdaI+M",J+m,scaled_identity+m), ("M-lambdaI",m-J,m-scaled_identity))
            residual = norm(actual-oracle)
            @printf("lambda=%g op=%s diag=%s expected=%s coefficient_residual=%.16g\n", scale,label,string(diag(NNF.jacobian(actual))),string(diag(NNF.jacobian(oracle))),residual)
            @assert residual > .5
            @assert norm((m+one(m))-(m+I)) == 0
        end
    end

    original = DAMap(nv=2,v=[x,p+.25x*x])
    println("unit_diagonal_map_direct_M_minus_3I_diag=",diag(NNF.jacobian(original-3I))," M_plus_negative_3I_diag=",diag(NNF.jacobian(original+(-3I)))," both_should_equal=",[-2.,-2.])
    @assert diag(NNF.jacobian(original-3I)) == [0.,0.]
    @assert diag(NNF.jacobian(original+(-3I))) == [2.,2.]

    println("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | NonlinearNormalForm 0.5.3 (`7e4053985736`), pinned clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | NonlinearNormalForm 0.5.3 (`7e4053985736`), pinned clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64, serial |

