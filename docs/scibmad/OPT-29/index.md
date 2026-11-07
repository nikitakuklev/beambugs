---
title: "OPT-29: ForwardDiff of a four-potential strength fails in implicit tracking"
hide:
  - toc
---

# OPT-29 — ForwardDiff of a four-potential strength fails in implicit tracking

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [`src/kernels/implicit.jl:138-147`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/implicit.jl#L138-L147)<br>[`:228-236`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/implicit.jl#L228-L236)<br>[`:243`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/implicit.jl#L243)<br>[`SciBmad.jl/README.md:24`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/README.md#L24)<br>[`:34`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/README.md#L34)<br>[`Beamlines.jl/src/misc.jl:97-113`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/misc.jl#L97-L113) |

Differentiating with ForwardDiff with respect to a strength parameter of a four-potential field tracked by BeamTracking's implicit integrator throws `MethodError: no method matching Float64(::ForwardDiff.Dual…)` in both fixed-point and Newton modes, although SciBmad advertises differentiable tracking through arbitrary electromagnetic fields. For the normalized potential A_s = −kx, Hamilton's equations give an exit derivative ∂pₓ/∂k = −L. A proton with reference momentum P₀c = 4mc²/3 through L = 0.55 m at k = 0.037 m⁻¹ should give −0.5500 m, and finite differences of tracking agree within 4.47×10⁻¹³ m. Plain tracking and the derivative with respect to the incoming pₓ work for this field, while field-strength differentiation also fails in a direct call of the implicit kernel.

Run `julia OPT-29.jl`.

??? example "Reproducer: OPT-29.jl"

    [Download OPT-29.jl](OPT-29.jl)

    ```julia
    # OPT-29: ForwardDiff of a four-potential strength fails in implicit tracking
    #
    # Differentiating with ForwardDiff with respect to a strength parameter of a four-potential field
    # tracked by BeamTracking's implicit integrator throws `MethodError: no method matching
    # Float64(::ForwardDiff.Dual…)` in both fixed-point and Newton modes, although SciBmad advertises
    # differentiable tracking through arbitrary electromagnetic fields. For the normalized potential A_s
    # = −kx, Hamilton's equations give an exit derivative ∂pₓ/∂k = −L. A proton with reference momentum
    # P₀c = 4mc²/3 through L = 0.55 m at k = 0.037 m⁻¹ should give −0.5500 m, and finite differences of
    # tracking agree within 4.47×10⁻¹³ m. Plain tracking and the derivative with respect to the incoming
    # pₓ work for this field, while field-strength differentiation also fails in a direct call of the
    # implicit kernel.
    #
    # The script tracks a proton with initial pₓ = 0.023 through 0.55 m of A_s = −kx in three
    # second-order implicit steps, in fixed-point and Newton modes, and checks the exit pₓ against pₓ −
    # kL, central differences at two step sizes against −L, and the ForwardDiff derivative with respect
    # to the incoming pₓ against 1. It then differentiates with respect to k through the lattice and
    # through a direct `implicit_step!` call, which excludes lattice promotion and builder shortcuts,
    # and prints each exception with its failing frame.
    using SciBmad, LinearAlgebra, Printf
    using SciBmad: BeamTracking, ForwardDiff
    const BT = BeamTracking
    const FD = ForwardDiff
    BLAS.set_num_threads(1)
    const SP = Species("proton")
    const R = (4/3)*massof(SP)/C_LIGHT
    const LEN = 0.55
    const K = 0.037
    const INITIAL = [0.013 0.023 -0.009 -0.014 0.025 0.04]

    # Uniform vertical magnetic field, in a gauge with only A_s nonzero.
    # H = -sqrt((eta+1/beta0)^2-mtilde^2-px^2-py^2) + k*x + eta/beta0.
    # Consequently dpx/ds = -k, exactly for both the ODE and its implicit split.
    function uniform_potential(x,y,s,t,p)
        z = zero(x)
        k = p[1]
        return (z,z,z,-k*x), ntuple(j -> j == 13 ? -k : z,16)
    end

    function solve_public(k,px_shift,newton)
        # Preallocate coordinates with the AD number type, required for mutating AD.
        b = Bunch(INITIAL .+ zero(k+px_shift);species=SP,p_over_q_ref=R)
        b.v[2] += px_shift
        e = LineElement(L=LEN,four_potential=uniform_potential,
            four_potential_params=(k,),four_potential_normalized=true,
            tracking_method=Symplectic(order=2,n_steps=3,implicit_use_newton=newton))
        track!(b,Beamline([e];species_ref=SP,p_over_q_ref=R);use_explicit_SIMD=false)
        @assert b.state[1] == BT.STATE_ALIVE
        return b.v[2]
    end

    function report_failure(f,label)
        try
            answer = f()
            println(label," unexpectedly succeeded: ",answer)
            return false
        catch err
            println(label," exception_type=",typeof(err))
            println("error_first_line=",first(split(sprint(showerror,err),'\n')))
            for frame in stacktrace(catch_backtrace())
                if occursin("implicit.jl",String(frame.file))
                    println("tracking_frame=",frame.func," ",basename(String(frame.file)),":",frame.line)
                end
            end
            @assert err isa MethodError
            return true
        end
    end

    for newton in (false,true)
        numeric = solve_public(K,0.0,newton)
        exact = INITIAL[2]-K*LEN
        @assert abs(numeric-exact) < 5e-16
        @printf("newton=%s numeric_px=%.17g exact_px=%.17g residual=%.3g\n",newton,numeric,exact,abs(numeric-exact))
        for h in (2e-5,1e-6)
            finite_difference = (solve_public(K+h,0.0,newton)-solve_public(K-h,0.0,newton))/(2*h)
            @assert abs(finite_difference+LEN) < 2e-12
            @printf("newton=%s finite_difference_h=%.1g derivative=%.17g exact=%.17g error=%.3g\n",
                    newton,h,finite_difference,-LEN,abs(finite_difference+LEN))
        end
        input_derivative = FD.derivative(a -> solve_public(K,a,newton),0.0)
        @assert abs(input_derivative-1) < 5e-15
        println("newton=",newton," coordinate_AD_dpx_dpxin=",input_derivative)
        @assert report_failure(() -> FD.derivative(a -> solve_public(a,0.0,newton),K),
                               "newton=$newton field_strength_AD")
    end

    # Direct kernel recheck excludes lattice promotion, BatchParam and builder shortcuts.
    const MT, GS, B0 = BT.drift_params(SP,R)
    function direct_step(k)
        v = INITIAL .+ zero(k)
        v[5] /= (1+INITIAL[6])/sqrt((1+INITIAL[6])^2+MT^2)
        v[6] = sqrt((1+INITIAL[6])^2+MT^2)-1/B0
        b = Bunch(v;species=SP,p_over_q_ref=R)
        BT.implicit_step!(1,b.coords,LEN/2,B0,MT,0.0,uniform_potential,(k,),R,Val(true),Val(false),LEN)
        return b.v[2]
    end
    @assert abs(direct_step(K)-(INITIAL[2]-K*LEN)) < 5e-16
    @assert report_failure(() -> FD.derivative(direct_step,K),"direct_kernel_strength_AD")
    println("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

