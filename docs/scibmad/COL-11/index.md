---
title: "COL-11: Sagan RF energy kicks double the leading off-axis spin rotation"
hide:
  - toc
---

# COL-11 — Sagan RF energy kicks double the leading off-axis spin rotation

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [`BeamTracking.jl/src/kernels/sagan_cavity.jl:193-198,215-216`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L193-L216)<br>[`SciBmad.jl/docs/src/sagancavity-physics.md:104-107`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/sagancavity-physics.md#L104-L107) |

The `SaganCavity` RF energy kick applies two full-strength electric spin rotations, doubling the leading spin rotation of particles with transverse momentum. An independent high-precision integral of the Thomas–BMT spin equation requires the single integrated electric impulse, or half-strength rotations if it is split around the acceleration. For a proton with P₀c, the reference momentum times the speed of light, equal to three rest energies and normalized horizontal momentum 0.1300, a 2000 eV energy gain in a zero-length cavity rotates the spin by −1.136 × 10⁻⁶ rad instead of −5.682 × 10⁻⁷ rad. Public tracking agrees with the kernel, and an on-axis particle is not rotated; exact doubling is established in the weak-gain limit, not for arbitrary finite gains.

Run `julia COL-11.jl`.

??? example "Reproducer: COL-11.jl"

    [Download COL-11.jl](COL-11.jl)

    ```julia
    # COL-11: Sagan RF energy kicks double the leading off-axis spin rotation
    #
    # The `SaganCavity` RF energy kick applies two full-strength electric spin rotations, doubling the
    # leading spin rotation of particles with transverse momentum. An independent high-precision
    # integral of the Thomas–BMT spin equation requires the single integrated electric impulse, or
    # half-strength rotations if it is split around the acceleration. For a proton with P₀c, the
    # reference momentum times the speed of light, equal to three rest energies and normalized
    # horizontal momentum 0.1300, a 2000 eV energy gain in a zero-length cavity rotates the spin by
    # −1.136 × 10⁻⁶ rad instead of −5.682 × 10⁻⁷ rad. Public tracking agrees with the kernel, and an
    # on-axis particle is not rotated; exact doubling is established in the weak-gain limit, not for
    # arbitrary finite gains.
    #
    # The script applies the Sagan RF energy kick directly to a proton with normalized horizontal
    # momentum 0.13 for gains of 2000 eV and 2 eV, and compares the spin angle with the closed-form
    # Thomas–BMT integral evaluated with 256-bit precision. It also tracks the same particle through a
    # zero-length `RFCavity` with `SaganCavity()`, which must match the direct kernel call, and checks
    # that an on-axis particle's spin is unchanged.
    using SciBmad, LinearAlgebra, Printf
    const BT = SciBmad.BeamTracking
    BLAS.set_num_threads(1)
    sp=Species("proton"); mass=massof(sp); pc=3mass; anomaly=gyromagnetic_anomaly(sp)
    checks = Bool[]
    px=0.13
    # Independent exact integration: gamma=A*cosh(u), where A^2=1+(Px/mc)^2.
    # Integral [a+1/(gamma+1)]/sqrt(gamma^2-A^2) dgamma
    # = a*u + 2/sqrt(A^2-1)*atan(sqrt((A-1)/(A+1))*tanh(u/2)).
    function analytic_rf_angle(gain)
        setprecision(BigFloat,256) do
            mt=BigFloat(mass)/BigFloat(pc)
            transverse=BigFloat(px)/mt
            A=sqrt(1+transverse^2)
            g0=sqrt(BigFloat(pc)^2+BigFloat(mass)^2)/BigFloat(mass)
            g1=g0+BigFloat(gain)/BigFloat(mass)
            F(g)=begin
                u=acosh(g/A)
                BigFloat(anomaly)*u+2/sqrt(A^2-1)*atan(sqrt((A-1)/(A+1))*tanh(u/2))
            end
            Float64(-transverse*(F(g1)-F(g0)))
        end
    end
    # One energy gain dE supplies one integrated longitudinal electric field. Finite gains can also
    # have endpoint-quadrature error, so the factor 2 is exact only in the weak-gain limit.
    rfangle(b)=2atan(b.coords.q[1,3],b.coords.q[1,1])
    println("longitudinal electric spin kick versus the exact BMT integral")
    for gain in (2000.,2.)
        exact=analytic_rf_angle(gain)
        beam=Bunch(v=[0. px 0 0 0 0],spin=true,species=sp,p_over_q_ref=pc/C_LIGHT)
        BT.sagan_cavity_kick!(1,beam.coords,anomaly,gain,2pi*7e7,0.,0.,mass,pc)
        angle=rfangle(beam)
        @printf("gain=%.1f kernel angle=%.17g analytic=%.17g ratio=%.17g\n",gain,angle,exact,angle/exact)
        push!(checks, abs(angle/exact-2) < 4e-12)
        line=Branch([Beamline([Marker()],species_ref=sp,pc_ref=pc),Beamline([RFCavity(L=0.,voltage=gain,rf_frequency=7e7,tracking_method=SaganCavity())],dE_ref=0.)]).beamlines[2]
        public_beam=Bunch(v=[0. px 0 0 0 0],spin=true,species=sp,p_over_q_ref=pc/C_LIGHT)
        track!(public_beam,line;use_explicit_SIMD=false)
        @printf("public angle=%.17g public/kernel residual=%.3g\n",rfangle(public_beam),rfangle(public_beam)-angle)
        @assert abs(rfangle(public_beam)-angle) < 1e-23
    end
    # Zero transverse momentum must give zero longitudinal-electric spin kick.
    axis=Bunch(v=zeros(1,6),spin=true,species=sp,p_over_q_ref=pc/C_LIGHT)
    BT.sagan_cavity_kick!(1,axis.coords,anomaly,2000.,2pi*7e7,0.,0.,mass,pc)
    @assert axis.coords.q[1,:] == [1.,0.,0.,0.]
    println("independent: public/kernel agreement and on-axis control pass")

    println(all(checks) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

