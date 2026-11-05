---
title: "OPT-26: TPSAMap composition transports covariance at the wrong expansion point"
hide:
  - toc
---

# OPT-26 — `TPSAMap` composition transports covariance at the wrong expansion point

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on NonlinearNormalForm main (`7e40539`) |
| Source | [`src/map.jl:464-468`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L464-L468)<br>[`src/map.jl:525-530`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L525-L530)<br>[`map.jl:15-19`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L15-L19)<br>[`:93-104`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/map.jl#L93-L104) |

When NonlinearNormalForm composes stochastic `TPSAMap`s, which carry a noise covariance S, it transports the incoming covariance with the second map's Jacobian at that map's own expansion point instead of at the incoming orbit. First-order transport requires S₂₁ = J₂S₁J₂ᵀ + S₂, with the Jacobian J₂ evaluated at the first map's output. Translating x by 0.75 with S₁ = diag(4, 9)×10⁻¹² and then applying the shear (x, p) → (x, p + x²) returns diag(4, 9)×10⁻¹² instead of [4 6; 6 18]×10⁻¹². Re-expanding the identical shear about the incoming orbit restores the expected covariance, and an exact finite-distribution calculation agrees with it to 3.029×10⁻²⁷. Matched expansion points pass.

??? example "Reproducer: OPT-26.jl"

    [Download OPT-26.jl](OPT-26.jl)

    ```julia
    # OPT-26: `TPSAMap` composition transports covariance at the wrong expansion point
    #
    # When NonlinearNormalForm composes stochastic `TPSAMap`s, which carry a noise covariance S, it
    # transports the incoming covariance with the second map's Jacobian at that map's own expansion
    # point instead of at the incoming orbit. First-order transport requires S₂₁ = J₂S₁J₂ᵀ + S₂, with
    # the Jacobian J₂ evaluated at the first map's output. Translating x by 0.75 with S₁ = diag(4,
    # 9)×10⁻¹² and then applying the shear (x, p) → (x, p + x²) returns diag(4, 9)×10⁻¹² instead of [4
    # 6; 6 18]×10⁻¹². Re-expanding the identical shear about the incoming orbit restores the expected
    # covariance, and an exact finite-distribution calculation agrees with it to 3.029×10⁻²⁷. Matched
    # expansion points pass.
    #
    # The script translates x by b = 0.75 with output noise S₁ on `Descriptor(2, 3)`, applies the exact
    # canonical shear, and compares the composed covariance with J₂S₁J₂ᵀ for J₂ = [1 0; 2b 1]. The same
    # shear expanded about (b, 0) must give that value with identical deterministic polynomials, and
    # four equiprobable noise points pushed through the shear in BigFloat arithmetic give the exact
    # covariance, because the shear's quadratic noise term is the same at all four. Composition at
    # matched expansion points is a passing control.
    using SciBmad, LinearAlgebra
    const NNF = NonlinearNormalForm
    BLAS.set_num_threads(1)
    x,p = vars(Descriptor(2,3))
    checks = Bool[]
    b = 0.75
    S = [4e-12 0.; 0. 9e-12]
    translation = NNF.TPSAMap(nv=2, v=[b+x,p], s=S)
    shear = NNF.TPSAMap(nv=2, v=[x,p+x^2], s=zeros(2,2))
    centered = NNF.TPSAMap(nv=2, v0=[b,0.], v=[b+x,p+(b+x)^2], s=zeros(2,2))
    uncentered_result = shear ∘ translation
    centered_result = centered ∘ translation
    # Jacobian of the shear at the incoming orbit (b, 0); the shear adds no noise, so first-order
    # transport gives S21 = J S1 Jᵀ.
    J = [1. 0.; 2b 1.]
    expected = J*S*transpose(J)
    # Independent exact nonlinear pushforward of two independent symmetric signs.
    sx,sp = sqrt(BigFloat(S[1,1])),sqrt(BigFloat(S[2,2]))
    # With dx = ±√Sxx the shear's dx² term is equal at all four points, so their covariance equals J S1
    # Jᵀ exactly, at finite noise.
    points = hcat(([b+u*sx, v*sp+(b+u*sx)^2] for u in (-1,1), v in (-1,1))...)
    mean_point = sum(points,dims=2)/4
    D = points .- mean_point
    finite_cov = D*transpose(D)/4
    println("observed / 1e-12 = ",Matrix(uncentered_result.s)/1e-12)
    println("expected / 1e-12 = ",expected/1e-12)
    println("recentered / 1e-12 = ",Matrix(centered_result.s)/1e-12)
    println("finite-distribution covariance / 1e-12 = ",Float64.(finite_cov)/1e-12)
    println("deterministic polynomial difference = ",maximum(GTPSA.normTPS.(uncentered_result.v .- centered_result.v)))
    @assert maximum(abs.(finite_cov-expected)) < 1e-26 # Exact nonlinear distribution, high precision.
    @assert maximum(abs.(Matrix(centered_result.s)-expected)) < 1e-26
    @assert maximum(GTPSA.normTPS.(uncentered_result.v .- centered_result.v)) == 0
    push!(checks, maximum(abs.(Matrix(uncentered_result.s)-expected)) > 8e-12)
    # Matched expansion points are a passing control.
    matched = NNF.TPSAMap(nv=2,v=[x,p],s=S)
    control = shear ∘ matched
    @assert Matrix(control.s) == S
    println("independent: finite-distribution covariance residual = ", Float64(maximum(abs.(finite_cov-expected))))
    println("matched-point control passes")

    println(all(checks) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | NonlinearNormalForm 0.5.3 (`7e4053985736`), source, Julia 1.13.2, CPU, Float64, serial |

