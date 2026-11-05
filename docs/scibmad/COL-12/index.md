---
title: "COL-12: Sextupole spin fringes use twice the longitudinal field"
hide:
  - toc
---

# COL-12 — Sextupole spin fringes use twice the longitudinal field

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking main (`0bf3585`) |
| Source | [`BeamTracking.jl/src/kernels/fringe.jl:115-120`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/fringe.jl#L115-L120)<br>[`219-224`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/fringe.jl#L219-L224)<br>[`BeamTracking.jl/src/kernels/multipole.jl:44`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/multipole.jl#L44) |

Normal and skew sextupole fringes apply twice the leading longitudinal-field spin rotation at both the entrance and the exit. Maxwell's equations fix the integrated longitudinal fringe field from the body field with its factorial normalization, and the gradient of an independently constructed scalar potential matches that body field. For a proton at x = 80.00 mm, y = 130.0 mm with zero transverse momentum, the derivative of the entrance spin angle with respect to the normalized normal-sextupole strength is −2.784 × 10⁻⁴ rad·m³ instead of −1.392 × 10⁻⁴ rad·m³. All four normal/skew and entrance/exit derivative checks give a factor 2.000.

??? example "Reproducer: COL-12.jl"

    [Download COL-12.jl](COL-12.jl)

    ```julia
    # COL-12: Sextupole spin fringes use twice the longitudinal field
    #
    # Normal and skew sextupole fringes apply twice the leading longitudinal-field spin rotation at both
    # the entrance and the exit. Maxwell's equations fix the integrated longitudinal fringe field from
    # the body field with its factorial normalization, and the gradient of an independently constructed
    # scalar potential matches that body field. For a proton at x = 80.00 mm, y = 130.0 mm with zero
    # transverse momentum, the derivative of the entrance spin angle with respect to the normalized
    # normal-sextupole strength is −2.784 × 10⁻⁴ rad·m³ instead of −1.392 × 10⁻⁴ rad·m³. All four
    # normal/skew and entrance/exit derivative checks give a factor 2.000.
    #
    # The script calls the straight-magnet `fringe!` kernel for a proton at x = 0.08 m, y = 0.13 m with
    # zero transverse momentum, for normal and skew sextupole strengths at the entrance and the exit,
    # and takes central derivatives of the spin angle with respect to strength (±10⁻⁷) to remove the
    # orbital-fringe correction. It compares each derivative with the Maxwell value −edge·(1 + a)·Ψ,
    # where Ψ is the scalar potential, a the anomaly and edge is +1 at the entrance and −1 at the exit,
    # and checks that the gradient of Ψ matches the package's body field.
    using SciBmad, LinearAlgebra, Printf
    const BT = SciBmad.BeamTracking
    BLAS.set_num_threads(1)
    sp=Species("proton"); mass=massof(sp); pc=3mass; anomaly=gyromagnetic_anomaly(sp)
    checks = Bool[]
    xx,yy=0.08,0.13
    # Body b_y+i*b_x=(Kn2+i*Ks2)*(x+i*y)^2/2!.
    # A scalar potential Psi has gradient (b_x,b_y), hence an entrance step
    # has integral b_s ds=Psi. These definitions fix the factorial independently.
    psi(kn,ks)=kn*(xx^2*yy-yy^3/3)/2 + ks*(xx^3/3-xx*yy^2)/2
    function fringe_angle(kn,ks,edge)
        b=Bunch(v=[xx 0. yy 0. 0. 0.],spin=true,species=sp,p_over_q_ref=pc/C_LIGHT)
        BT.fringe!(1,b.coords,anomaly,mass/pc,nothing,nothing,nothing,nothing,nothing,kn,ks,Val(edge))
        2atan(b.coords.q[1,4],b.coords.q[1,1])
    end
    for (kn,ks) in ((1.,0.),(0.,1.)), edge in (1,-1)
        epsk=1e-7
        # Central strength derivative removes leading changes in fringe orbit.
        derivative=(fringe_angle(epsk*kn,epsk*ks,edge)-fringe_angle(-epsk*kn,-epsk*ks,edge))/(2epsk)
        # Integral b_s ds = edge*Psi; for zero transverse momentum the leading BMT angle is
        # -edge*(1+anomaly)*Psi.
        expected=-edge*(1+anomaly)*psi(kn,ks)
        @printf("normal=%.0f skew=%.0f edge=%d spin derivative=%.17g Maxwell=%.17g ratio=%.17g\n",kn,ks,edge,derivative,expected,derivative/expected)
        push!(checks, abs(derivative/expected-2) < 3e-14)
        bx,by=BT.normalized_field(BT.SA[3],(kn,),(ks,),xx,yy,0)
        dxpsi=kn*xx*yy+ks*(xx^2-yy^2)/2
        dypsi=kn*(xx^2-yy^2)/2-ks*xx*yy
        @assert abs(bx-dxpsi) < 1e-17 && abs(by-dypsi) < 1e-17
    end
    println("independent: body field equals the scalar-potential gradient at both edges for normal/skew sextupoles")

    println(all(checks) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

