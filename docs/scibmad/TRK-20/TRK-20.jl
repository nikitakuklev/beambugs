# TRK-20: Zero alignment applies a bend's reference roll twice
#
# Adding an explicit zero `AlignmentParams` group to a bend with a reference roll changes its orbit;
# at zero reference curvature the roll is applied twice. Zero alignment must preserve transport, and
# an independent Lorentz-force trajectory agrees with the unaligned map at the intended roll and
# with the aligned map at twice that roll. For a 0.1900 m straight-reference bend with normalized
# strength −0.1700 m⁻¹ and reference roll 0.01500 rad, the kick in normalized vertical momentum is
# 9.689 × 10⁻⁴ instead of 4.845 × 10⁻⁴. This occurs with `Exact`, `Symplectic` and the default
# method, and bends with finite curvature also change under zero alignment; omitting or ignoring the
# alignment group restores the unaligned map.
#
# The script tracks four protons with nonzero momenta through straight-reference bends with rolls
# −0.63, 0.015, 0 and 1.1 rad, strengths −0.17 and 0.11 m⁻¹ and lengths 0.19 and 0.63 m, using
# `Exact` and fourth-order `Symplectic` with fringes off, with and without SIMD. It compares the
# maps without alignment and with an explicit zero `x_offset` against an independent Lorentz-force
# orbit at the roll a and at 2a. Further cases cover curved bends, where ignoring the alignment
# group is the control, a finite alignment tilt, and the default method.
using SciBmad, LinearAlgebra, Printf
BLAS.set_num_threads(1)
const sp = Species("proton")
const vin = [0.002 0.013 -0.001 -0.009 0.004 -0.21;
             -0.003 -0.017 0.004 0.021 -0.003 0.18;
             0.001 0.0 0.002 0.0 0.0 0.0;
             -0.002 0.009 -0.003 0.012 0.002 0.37]

function transport(v, a, g, k, L, method, alignment; simd=false, pc=1.7e9)
    kw = alignment == :absent ? (;) : alignment == :zero ? (;x_offset=0.0) :
         alignment == :tiny ? (;x_offset=1e-10) : alignment == :ignored ?
         (;x_offset=0.0,ignore_params=[AlignmentParams]) : alignment == :tilted ?
         (;tilt=0.023) : error("Unknown alignment case: $alignment")
    e = SBend(;L,g_ref=g,Kn0=k,tilt_ref=a,tracking_method=method,kw...)
    b = Bunch(copy(v),species=sp,p_over_q_ref=pc/C_LIGHT)
    track!(b,Beamline([e],species_ref=sp,pc_ref=pc);use_explicit_SIMD=simd)
    @assert all(b.state .== BeamTracking.STATE_ALIVE)
    return copy(b.v)
end

# Independent uniform-field orbit: integrate q'= -k, u'=0 in a Cartesian
# longitudinal frame, then rotate the displacement back into x and y.
function orbit(v, a, k, L)
    c,s = cos(a),sin(a)
    out = copy(v[:,1:4])
    for i in axes(v,1)
        q0 = c*v[i,2]+s*v[i,4]
        u = -s*v[i,2]+c*v[i,4]
        A = sqrt((1+v[i,6])^2-u^2)
        q1 = q0-k*L
        dq = (sqrt(A^2-q1^2)-sqrt(A^2-q0^2))/k
        du = u*(asin(q0/A)-asin(q1/A))/k
        out[i,1] += c*dq-s*du
        out[i,3] += s*dq+c*du
        out[i,2] -= k*L*c
        out[i,4] -= k*L*s
    end
    out
end

function rotate(v,a)
    w=copy(v); c,s=cos(a),sin(a)
    for (i,j) in ((1,3),(2,4))
        w[:,i] = c*v[:,i]-s*v[:,j]
        w[:,j] = s*v[:,i]+c*v[:,j]
    end
    w
end

# Residual maxima over all six coordinates mix positions (m) and dimensionless normalized momenta
# and are component diagnostics. Exact roll doubling is claimed only at zero reference curvature;
# curved cases establish only that zero alignment changes the map.
function main()
    methods=(Exact(fringe_at=Fringe.NoEnd),Symplectic(order=4,n_steps=3,fringe_at=Fringe.NoEnd))
    for method in methods, simd in (false,true)
        worst_clean=0.0; worst_double=0.0; worst_shift=0.0; worst_identity=0.0
        for a in (-0.63,0.015,0.0,1.1), k in (-0.17,0.11), L in (0.19,0.63)
            clean=transport(vin,a,0.0,k,L,method,:absent;simd)
            zeroa=transport(vin,a,0.0,k,L,method,:zero;simd)
            tiny=transport(vin,a,0.0,k,L,method,:tiny;simd)
            worst_clean=max(worst_clean,maximum(abs,clean[:,1:4]-orbit(vin,a,k,L)))
            worst_double=max(worst_double,maximum(abs,zeroa[:,1:4]-orbit(vin,2a,k,L)))
            worst_shift=max(worst_shift,maximum(abs,tiny-zeroa))
            worst_identity=max(worst_identity,maximum(abs,zeroa-clean))
            @assert maximum(abs,clean[:,1:4]-orbit(vin,a,k,L)) < 2e-14
            @assert maximum(abs,zeroa[:,1:4]-orbit(vin,2a,k,L)) < 2e-14
            @assert maximum(abs,tiny-zeroa) < 2e-14
            if a==0; @assert maximum(abs,zeroa-clean)<2e-14; end
        end
        @printf("SWEEP method=%s simd=%s cases=16 clean_orbit=%.17g double_orbit=%.17g tiny_shift=%.17g zero_identity=%.17g\n",nameof(typeof(method)),simd,worst_clean,worst_double,worst_shift,worst_identity)
        for (a,k,L) in ((-0.63,0.11,0.63),(0.015,-0.17,0.19))
            out=transport(vin,a,0.0,k,L,method,:zero;simd)
            kick=out[3,[2,4]]-vin[3,[2,4]]
            # In a straight reference frame the roll enters once: (dpx, dpy) = -k L (cos a, sin a).
            # A doubled roll misses this kick vector by 2|k L sin(a/2)|.
            pred=-k*L*[cos(a),sin(a)]
            @printf("KICK a=%.6g k=%.6g L=%.6g expected=(%.17g,%.17g) actual=(%.17g,%.17g) norm_error=%.17g analytic_error=%.17g\n",a,k,L,pred...,kick...,norm(kick-pred),2abs(k*L*sin(a/2)))
        end
    end
    method=first(methods)
    for g in (-0.12,0.17), a in (-0.63,0.28), pc in (1.7e9,4.3e9)
        clean=transport(vin,a,g,0.14,0.43,method,:absent;pc)
        aligned=transport(vin,a,g,0.14,0.43,method,:zero;pc)
        ignored=transport(vin,a,g,0.14,0.43,method,:ignored;pc)
        reference=rotate(transport(rotate(vin,-a),0.0,g,0.14,0.43,method,:absent;pc),a)
        err=maximum(abs,clean-reference)
        @printf("CURVED g=%.6g a=%.6g pc=%.6g zero_identity=%.17g covariance=%.17g ignore_control=%.17g\n",g,a,pc,maximum(abs,aligned-clean),err,maximum(abs,ignored-clean))
        @assert err<3e-14
        @assert maximum(abs,ignored-clean)<3e-14
        @assert maximum(abs,aligned-clean)>1e-4
    end
    # A nonzero physical alignment tilt must add to the single reference roll.
    a=-0.63; k=0.11; L=0.63
    out=transport(vin,a,0.0,k,L,method,:tilted)
    correct=orbit(vin,a+0.023,k,L)
    doubled=orbit(vin,2a+0.023,k,L)
    @printf("FINITE_ALIGNMENT correct_orbit_error=%.17g double_orbit_error=%.17g\n",maximum(abs,out[:,1:4]-correct),maximum(abs,out[:,1:4]-doubled))
    @assert maximum(abs,out[:,1:4]-doubled)<2e-14
    # The public default takes the same faulty path with a zero alignment.
    default=transport(vin,a,0.0,k,L,SciBmadStandard(),:zero)
    default_clean=transport(vin,a,0.0,k,L,SciBmadStandard(),:absent)
    default_twice=transport(vin,2a,0.0,k,L,SciBmadStandard(),:absent)
    @printf("DEFAULT zero_identity=%.17g double_roll_map_error=%.17g\n",maximum(abs,default-default_clean),maximum(abs,default-default_twice))
    @assert maximum(abs,default-default_twice)<3e-14
    println("DEFECT REPRODUCED")
end
main()
