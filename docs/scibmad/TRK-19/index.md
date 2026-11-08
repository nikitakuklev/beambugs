---
title: "TRK-19: Callback out/in transforms change momentum in a time-dependent potential"
hide:
  - toc
---

# TRK-19 — Callback out/in transforms change momentum in a time-dependent potential

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [src/kernels/implicit.jl:789–807](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/implicit.jl#L789-L807)<br>[src/kernels/implicit.jl:792](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/implicit.jl#L792)<br>[docs/src/track.md:178–203](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/track.md#L178-L203)<br>[src/kernels/rfcavity_kick.jl:47–98](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/rfcavity_kick.jl#L47-L98) |

A callback whose coordinate transforms only take particles out for observation and back in changes their momentum in a time-dependent scalar potential. The observer contract requires an unchanged trajectory, and an independent mass-shell calculation identifies an incorrect time in the inward conversion. For a proton in a 0.6000 m element with dimensionless potential φ = κct·16(s/L)²(1 − s/L)², where κ = 0.3000 m⁻¹, c is light speed, t is time, s is position and L is the element length, the final relative momentum offset is 0.1993 instead of 0.1835. Constant-potential and no-op-callback controls agree with tracking without an observer; the error requires both the coordinate transforms and time dependence.

Run `julia TRK-19.jl`.

??? example "Reproducer: TRK-19.jl"

    [Download TRK-19.jl](TRK-19.jl)

    ```julia
    # TRK-19: Callback out/in transforms change momentum in a time-dependent potential
    #
    # A callback whose coordinate transforms only take particles out for observation and back in changes
    # their momentum in a time-dependent scalar potential. The observer contract requires an unchanged
    # trajectory, and an independent mass-shell calculation identifies an incorrect time in the inward
    # conversion. For a proton in a 0.6000 m element with dimensionless potential φ = κct·16(s/L)²(1 −
    # s/L)², where κ = 0.3000 m⁻¹, c is light speed, t is time, s is position and L is the element
    # length, the final relative momentum offset is 0.1993 instead of 0.1835. Constant-potential and
    # no-op-callback controls agree with tracking without an observer; the error requires both the
    # coordinate transforms and time dependence.
    #
    # The script first applies the callback's out and in transforms directly to a proton with p₀c =
    # 4mc²/3 at s = 0.21 m in the potential φ = 0.017 + κct, for normalized and unnormalized inputs; it
    # builds the internal energy coordinate from the mass shell and compares its change with the
    # predicted κ(τ − z), where τ = z/β, and checks that re-entering with the correct time restores
    # every coordinate. It then tracks through the 0.6 m element above in five second-order steps with
    # no callback, a no-op callback, and an observer that applies the out and in transforms, at κ = 0
    # and 0.3 m⁻¹.
    using SciBmad
    using SciBmad: BeamTracking
    using LinearAlgebra, Printf
    const BT = BeamTracking
    BLAS.set_num_threads(1)
    const SP = Species("proton")
    const R = (4/3)*massof(SP)/C_LIGHT
    const MT, GS, B0 = BT.drift_params(SP, R)
    const LAB = [0.012 0.025 -0.008 -0.015 0.07 0.04]

    # Returned scalar potential is divided by c inside the integrator.
    function linear_time_potential(x, y, s, t, p)
        k, offset, scale = p
        a = scale*C_LIGHT*(offset + k*C_LIGHT*t)
        z = zero(x)
        return (a,z,z,z), ntuple(j -> j == 4 ? scale*k*C_LIGHT^2 : z, 16)
    end

    function roundtrip(k, normalized)
        s = 0.21
        offset = 0.017
        beta = (1+LAB[6])/sqrt((1+LAB[6])^2 + MT^2)
        tau = LAB[5]/beta
        phi = offset + k*(s/B0 - tau)
        # Independent physical construction, rather than seeding via upstream conversion.
        eta = sqrt((1+LAB[6])^2 + MT^2) - 1/B0 + phi
        internal = copy(LAB)
        internal[5] = tau
        internal[6] = eta
        b = Bunch(copy(internal); species=SP, p_over_q_ref=R)
        scale = normalized ? 1.0 : R
        args = (s, s/(B0*C_LIGHT), B0, MT, linear_time_potential,
                (k,offset,scale), R, Val(normalized))
        BT.callback_implicit!(1,b.coords,args...,Val(false))
        outside = copy(b.v)
        @assert maximum(abs, outside-LAB) < 5e-16
        BT.callback_implicit!(1,b.coords,args...,Val(true))
        # callback_implicit! evaluates the potential at time (s/beta0 - v5)/c in both directions, but v5
        # is tau = z/beta on the way out and z on the way in, so the inward conversion uses the wrong
        # time and the energy coordinate changes by kappa*(tau - z).
        measured = b.v[6]-eta
        expected = k*(tau-outside[5])
        @assert abs(measured-expected) < 2e-16
        @assert b.state[1] == BT.STATE_ALIVE
        # Re-enter using the physical time z/(beta*c).
        b.v .= outside
        BT.bmad_to_mad!(1,b.coords,B0,MT,phi)
        restored = maximum(abs,b.v-internal)
        @assert restored < 5e-16
        @printf("roundtrip k=%+.2f normalized=%s eta_error=%.17g analytic=%.17g residual=%.3g correct_time_maxerr=%.3g\n",
                k, normalized, measured, expected, abs(measured-expected), restored)
    end
    for normalized in (true,false), k in (0.0,0.15,-0.11)
        roundtrip(k,normalized)
    end

    # Smooth compact-element scalar potential; both potential and s derivative
    # vanish at the boundaries, avoiding a boundary/gauge-step interpretation.
    const LEN = 0.6
    function enveloped_potential(x,y,s,t,p)
        u = s/LEN
        h = 16*u^2*(1-u)^2
        dh = 32*u*(1-u)*(1-2*u)/LEN
        k = p[1]
        z = zero(x)
        a = k*C_LIGHT^2*t
        return (a*h,z,z,z), ntuple(j -> j == 3 ? a*dh : j == 4 ? k*C_LIGHT^2*h : z, 16)
    end
    function observer(i,c,s,t,bg,ds,g,outfun,infun)
        outfun(i,c,s,t)
        infun(i,c,s,t)
        return nothing
    end
    noop(i,c,s,t,bg,ds,g,outfun,infun) = nothing
    function transport(callbacks,k)
        b = Bunch(copy(LAB);species=SP,p_over_q_ref=R,callbacks)
        e = LineElement(L=LEN,four_potential=enveloped_potential,
            four_potential_params=(k,),four_potential_normalized=true,
            tracking_method=Symplectic(order=2,n_steps=5))
        line = Beamline([e];species_ref=SP,p_over_q_ref=R)
        track!(b,line;use_explicit_SIMD=false)
        @assert b.state[1] == BT.STATE_ALIVE
        return copy(b.v)
    end
    for k in (0.0,0.3)
        baseline = transport((),k)
        untouched = transport((noop,),k)
        observed = transport((observer,),k)
        @assert maximum(abs,baseline-untouched) < 5e-16
        if k == 0
            @assert maximum(abs,baseline-observed) < 5e-16
        else
            @assert abs(baseline[6]-observed[6]) > 1e-4
        end
        @printf("public k=%.1f no_callback_delta=%.17g noop_delta=%.17g observer_delta=%.17g alive=true\n",
                k,baseline[6],untouched[6],observed[6])
    end
    println("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

