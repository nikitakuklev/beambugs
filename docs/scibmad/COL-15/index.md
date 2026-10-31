---
title: "COL-15: IBS uses canonical momenta for flight time inside a solenoid"
hide:
  - toc
---

# COL-15 — IBS uses canonical momenta for flight time inside a solenoid

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking main (`0bf3585`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`BeamTracking.jl/src/kernels/ibs_kick.jl:22-35`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/ibs_kick.jl#L22-L35)<br>[`ext/BeamTrackingBeamlinesExt/general_bl.jl:98-167`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/general_bl.jl#L98-L167)<br>[`BeamTrackingBeamlinesExt.jl:12-55`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/BeamTrackingBeamlinesExt.jl#L12-L55)<br>[`unpack_bl.jl:1-55`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L1-L55)<br>[`unpack_bl.jl:156-160`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L156-L160)<br>[`unpack_bl.jl:267-274`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L267-L274)<br>[`538`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L538)<br>[`unpack_bl.jl:345`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L345)<br>[`symplectic_bl.jl:73-94`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L73-L94)<br>[`src/kernels/solenoid_kick.jl:62-65`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/solenoid_kick.jl#L62-L65)<br>[`src/types.jl:69`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/types.jl#L69)<br>[`docs/src/track.md:24-29`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/track.md#L24-L29)<br>[`src/tracking_methods.jl:8-45`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/tracking_methods.jl#L8-L45)<br>[`utils_bl.jl:230-244`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L230-L244)<br>[`src/kernels/integrators.jl:5-33`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/integrators.jl#L5-L33) |

Inside a solenoid, the intrabeam scattering (IBS) kick computes flight time and particle survival from canonical transverse momenta without subtracting the solenoid vector potential. A translation of the bunch that preserves mechanical momenta must preserve every collision kick, as independent Lorentz-helix tracking and the unchanged bunch covariance confirm. For 64 protons with reference momentum 3mc (m the proton mass, c the speed of light), normalized solenoid strength 8.000 m⁻¹ and slice length 7.813 mm, a 31.25 mm horizontal translation changes the largest impulse by 0.8438% of the unshifted value instead of zero. A 0.3750 m vertical translation falsely loses all 64 particles; zero-field controls pass, and source inspection shows that the stochastic kick amplitudes share the faulty flight time.

??? example "Reproducer: COL-15.jl"

    [Download COL-15.jl](COL-15.jl)

    ```julia
    # COL-15: IBS uses canonical momenta for flight time inside a solenoid
    #
    # Inside a solenoid, the intrabeam scattering (IBS) kick computes flight time and particle survival
    # from canonical transverse momenta without subtracting the solenoid vector potential. A translation
    # of the bunch that preserves mechanical momenta must preserve every collision kick, as independent
    # Lorentz-helix tracking and the unchanged bunch covariance confirm. For 64 protons with reference
    # momentum 3mc (m the proton mass, c the speed of light), normalized solenoid strength 8.000 m⁻¹ and
    # slice length 7.813 mm, a 31.25 mm horizontal translation changes the largest impulse by 0.8438% of
    # the unshifted value instead of zero. A 0.3750 m vertical translation falsely loses all 64
    # particles; zero-field controls pass, and source inspection shows that the stochastic kick
    # amplitudes share the faulty flight time.
    #
    # The script tracks bunches through an interior slice of a uniform solenoid with deterministic IBS
    # and compares the longitudinal impulses, measured as the change of the relative momentum deviation
    # δ that the solenoid itself conserves, before and after translations that preserve mechanical
    # momenta. It uses a 64-particle grid of protons and electrons at several solenoid strengths, a
    # 256-particle Gaussian sample and a separate 16-proton sample; large vertical translations show the
    # false losses, also at unit weight. Controls compare tracking without IBS with an independent
    # Lorentz-helix integration, check that zero field gives no translation error, and scale the length
    # and population weight.

    module IndependentChecks
    # Independent sample and Lorentz-equation oracle. Deterministic IBS is measured
    # in delta, which a static solenoid body conserves. No kick coefficients from
    # BeamTracking are used in the expected translation or helix.
    using SciBmad, LinearAlgebra, Statistics, Printf, Random
    BLAS.set_num_threads(1)
    const BT = SciBmad.BeamTracking

    function physical_sample(;gaussian=false)
        widths = [2.0^-10,2.0^-12,2.0^-11,2.0^-13,2.0^-6,2.0^-10]
        if gaussian
            return randn(MersenneTwister(271828),256,6).*permutedims(widths)
        end
        # All 2^6 vertices, rather than a selection of Walsh columns.
        return [(iszero((i >> (j-1)) & 1) ? -1 : 1)*widths[j] for i in 0:63,j in 1:6]
    end
    function canonical(physical,K,dx,dy)
        v = copy(physical)
        v[:,1] .+= dx
        v[:,3] .+= dy
        v[:,2] .-= K/2 .* v[:,3]
        v[:,4] .+= K/2 .* v[:,1]
        return v
    end
    function shells(v,K)
        r = 1 .+ v[:,6]
        u = v[:,2] .+ K/2 .* v[:,3]
        w = v[:,4] .- K/2 .* v[:,1]
        return r.^2-u.^2-w.^2, r.^2-v[:,2].^2-v[:,4].^2
    end
    function covariance(v)
        centered = v .- mean(v;dims=1)
        return centered'*centered/size(v,1)
    end
    function helix(v,K,L,bg)
        out = copy(v)
        beta0 = bg/sqrt(1+bg^2)
        for i in axes(v,1)
            x,px,y,py,z,delta = v[i,:]
            r = 1+delta
            u,w = px+K*y/2,py-K*x/2
            ps = sqrt(r^2-u^2-w^2)
            theta = K*L/ps
            sn,cs = sincos(theta)
            omc = 2*sin(theta/2)^2
            if K == 0
                X,Y = x+L*u/ps,y+L*w/ps
            else
                X,Y = x+(u*sn+w*omc)/K,y+(w*sn-u*omc)/K
            end
            U,W = u*cs+w*sn,w*cs-u*sn
            beta = r/sqrt(r^2+inv(bg)^2)
            out[i,:] = [X,U-K*Y/2,Y,W+K*X/2,z+L*(beta/beta0-r/ps),delta]
        end
        return out
    end
    function run(v,K,L,sp,bg,weight,simd;ibs=true,n_steps=1)
        tm = Symplectic(order=2,n_steps=n_steps,fringe_at=BT.Fringe.NoEnd,
            ibs_damping_on=ibs,ibs_fluctuations_on=false)
        line = Beamline([Solenoid(L=L,Ksol=K,tracking_method=tm)],
            species_ref=sp,pc_ref=bg*massof(sp))
        b = Bunch(v=copy(v),species=sp,p_over_q_ref=line.p_over_q_ref,weight=weight)
        @assert all(==(BT.STATE_ALIVE),b.coords.state)
        track!(b,line;use_explicit_SIMD=simd)
        return copy(b.coords.v),copy(b.coords.state)
    end

    function audit_sample(physical,sp,bg,K,L,weight,simd;gaussian=false)
        base = canonical(physical,K,0,0)
        @assert isposdef(Symmetric(covariance(base)))
        base_out,base_state = run(base,K,L,sp,bg,weight,simd)
        @assert all(==(BT.STATE_ALIVE),base_state)
        kick0 = base_out[:,6]-base[:,6]
        @assert maximum(abs,kick0) > 1e-12
        _,base_shell = shells(base,K)
        # Canonical translation preserves mechanical momenta and every IBS impulse; positive physical
        # longitudinal momentum must preserve survival.
        translations = ((2.0^-10,0.0),(1/32,0.0),(-1/8,1/16),(0.0,3/8))
        for (dx,dy) in translations
            v = canonical(physical,K,dx,dy)
            physical_shell,canonical_shell = shells(v,K)
            @assert minimum(physical_shell) > 0.99
            cov_err = maximum(abs,covariance(v)-covariance(base))
            @assert cov_err < 1e-18
            bare,bare_state = run(v,K,L,sp,bg,weight,simd;ibs=false)
            helix_error = maximum(abs,bare-helix(v,K,L,bg))
            @assert all(==(BT.STATE_ALIVE),bare_state)
            @assert helix_error < 9e-16
            out,state = run(v,K,L,sp,bg,weight,simd)
            alive = count(==(BT.STATE_ALIVE),state)
            if minimum(canonical_shell) > 0
                kick = out[:,6]-v[:,6]
                ratio = sqrt.(base_shell./canonical_shell)
                relative = maximum(abs,kick-kick0)/maximum(abs,kick0)
                prediction = maximum(abs,kick-ratio.*kick0)/maximum(abs,kick0)
                @assert alive == size(v,1)
                @assert prediction < 1e-9
                if K == 0
                    @assert relative < 1e-9
                elseif !gaussian && dx == 1/32
                    @assert relative > 0.007
                end
                @printf("TIMING species=%s bg=%.6g K=%.6g L=%.6g simd=%s gaussian=%s shift=(%.9g,%.9g) relative_error=%.17g timing_prediction_residual=%.17g min_physical_ps2=%.17g helix_error=%.17g covariance_error=%.17g kick0_1=%.17g shifted_kick_1=%.17g\n",
                    string(sp),bg,K,L,simd,gaussian,dx,dy,relative,prediction,minimum(physical_shell),helix_error,cov_err,kick0[1],kick[1])
            else
                @assert alive == 0
                # False loss occurs before any damping; unit population weight
                # independently excludes a collective-rate or large-kick cause.
                _,unit_state = run(v,K,L,sp,bg,1.0,simd)
                @assert count(==(BT.STATE_ALIVE),unit_state) == 0
                @printf("LOSS species=%s bg=%.6g K=%.6g simd=%s gaussian=%s shift=(%.9g,%.9g) expected_alive=%d actual_alive=%d unit_weight_alive=0 min_physical_ps2=%.17g min_canonical_ps2=%.17g helix_error=%.17g\n",
                    string(sp),bg,K,simd,gaussian,dx,dy,size(v,1),alive,minimum(physical_shell),minimum(canonical_shell),helix_error)
            end
        end
        return kick0
    end

    grid = physical_sample()
    for simd in (false,true), (name,bg,W) in (("proton",3.0,1e13),("electron",9.0,1e8)), K in (-8.0,8.0,16.0,0.0)
        audit_sample(grid,Species(name),bg,K,1/128,W,simd)
    end

    # Change population and element length independently. The full-length IBS
    # impulse must scale linearly in each, even when orbital integration is split.
    sp = Species("proton")
    K,L,bg,W = 8.0,1/128,3.0,1e13
    v = canonical(grid,K,1/32,0)
    nominal,_ = run(v,K,L,sp,bg,W,false)
    reference = nominal[:,6]-v[:,6]
    for (lf,wf,steps) in ((3.0,1.0,1),(1.0,0.25,1),(1.0,1.0,4))
        out,state = run(v,K,L*lf,sp,bg,W*wf,false;n_steps=steps)
        kick = out[:,6]-v[:,6]
        err = maximum(abs,kick-lf*wf*reference)/maximum(abs,lf*wf*reference)
        @assert err < 1e-9 && all(==(BT.STATE_ALIVE),state)
        println("SCALING length_factor=",lf," weight_factor=",wf," n_steps=",steps," relative_residual=",err)
    end

    # A second sample breaks the equal-radius tensor symmetry and changes N.
    audit_sample(physical_sample(gaussian=true),Species("proton"),3.0,8.0,1/128,1e13,false;gaussian=true)
    println("IndependentChecks PASS")
    end

    module DiscoveryChecks
    using SciBmad, LinearAlgebra, Printf
    BLAS.set_num_threads(1)
    const BT=BeamTracking
    const SP=Species("proton")
    const PC=2massof(SP)
    const R=PC/C_LIGHT
    const L=0.01
    const K=4.0
    const WEIGHT=1e12
    const SIG=[2.0^-10,2.0^-12,2.0^-11,2.0^-13,2.0^-8,2.0^-11]
    const MASKS=[1,2,4,8,3,5]

    # Six orthogonal nonconstant Walsh columns give a positive-definite covariance
    # using 16 equally weighted particles, with exactly representable coordinates.
    function initial(k)
        v=[(isodd(count_ones(i & MASKS[j])) ? -1.0 : 1.0)*SIG[j] for i=0:15,j=1:6]
        v[:,2].-=k/2*v[:,3]
        v[:,4].+=k/2*v[:,1]
        v
    end
    function translate_beam(v,dx,dy,k)
        w=copy(v)
        w[:,1].+=dx; w[:,3].+=dy
        w[:,2].-=k*dy/2; w[:,4].+=k*dx/2
        w
    end

    # Independent Lorentz helix in a uniform longitudinal magnetic field. In the
    # symmetric gauge, u=(px+Ky/2,py-Kx/2) is the mechanical transverse momentum.
    function helix(v,k,length)
        out=copy(v)
        for i=axes(v,1)
            x,px,y,py,z,delta=v[i,:]
            u=1+delta; ux=px+k*y/2; uy=py-k*x/2
            ps=sqrt(u^2-ux^2-uy^2)
            theta=k*length/ps
            a=iszero(k) ? length/ps : sin(theta)/k
            b=iszero(k) ? 0.0 : 2sin(theta/2)^2/k
            xf=x+a*ux+b*uy; yf=y+a*uy-b*ux
            uxf=cos(theta)*ux+sin(theta)*uy
            uyf=cos(theta)*uy-sin(theta)*ux
            beta=u/hypot(u,massof(SP)/PC); beta0=PC/hypot(PC,massof(SP))
            out[i,:]=[xf,uxf-k*yf/2,yf,uyf+k*xf/2,
                      z+length*beta/beta0-u*length/ps,delta]
        end
        out
    end
    function tracked(v,k,on,simd;weight=WEIGHT)
        b=Bunch(;v=copy(v),species=SP,p_over_q_ref=R,weight)
        tm=Symplectic(order=2,n_steps=1,fringe_at=Fringe.NoEnd,ibs_damping_on=on)
        bl=Beamline([Solenoid(;L,Ksol=k,tracking_method=tm)];species_ref=SP,pc_ref=PC)
        track!(b,bl;use_explicit_SIMD=simd,use_KA=false)
        b
    end
    function physical_ps2(v,k)
        (1 .+ v[:,6]).^2-(v[:,2]+k/2*v[:,3]).^2-(v[:,4]-k/2*v[:,1]).^2
    end
    function canonical_ps2(v)
        (1 .+ v[:,6]).^2-v[:,2].^2-v[:,4].^2
    end

    for simd in (false,true)
        v=initial(K)
        base=tracked(v,K,true,simd)
        @assert all(==(BT.STATE_ALIVE),base.state)
        # Undo the independently known body to recover the entry-plane IBS impulse.
        kick0=(helix(base.v,K,-L)-v)[:,[2,4,6]]
        println("MODE simd=",simd," particles=",size(v,1)," weight=",WEIGHT)
        println("BASE kick_norm=",norm(kick0)," max_abs=",maximum(abs,kick0))
        for (dx,dy) in ((0.0,0.0),(0.125,0.0),(0.25,0.0),(0.0,0.25),(0.75,0.0))
            w=translate_beam(v,dx,dy,K)
            physical_min=minimum(physical_ps2(w,K))
            @assert physical_min>0.99
            got=tracked(w,K,true,simd)
            off=tracked(w,K,false,simd)
            off_expected=helix(w,K,L)
            control_error=maximum(abs,off.v-off_expected)
            @assert all(==(BT.STATE_ALIVE),off.state) && control_error<2e-15
            # A homogeneous field and collisions are invariant under this magnetic
            # translation. The map should commute with the canonical translation.
            expected=translate_beam(base.v,dx,dy,K)
            map_error=maximum(abs,got.v-expected)
            println("CASE dx=",dx," dy=",dy," physical_ps2_min=",physical_min,
                    " canonical_ps2_min=",minimum(canonical_ps2(w)),
                    " expected_alive=16 actual_alive=",count(==(BT.STATE_ALIVE),got.state),
                    " IBS_off_helix_error=",control_error," translation_map_error=",map_error)
            if all(==(BT.STATE_ALIVE),got.state)
                d=(helix(got.v,K,-L)-w)[:,[2,4,6]]
                ratio=sqrt.(canonical_ps2(v)./canonical_ps2(w))
                rel=norm(d-kick0)/norm(kick0)
                ratio_resid=norm(d-kick0.*ratio)/norm(d)
                # Delta is unchanged by the magnetic body. This measurement avoids
                # subtracting the larger, translated canonical transverse momenta.
                delta0=base.v[:,6]-v[:,6]
                delta=got.v[:,6]-w[:,6]
                delta_rel=norm(delta-delta0)/norm(delta0)
                delta_ratio_resid=norm(delta-delta0.*ratio)/norm(delta)
                println("KICK relative_translation_error=",rel,
                        " source_timing_ratio_residual=",ratio_resid)
                println("DELTA relative_translation_error=",delta_rel,
                        " source_timing_ratio_residual=",delta_ratio_resid)
                for i=1:16
                    @printf("PARTICLE %d expected=(%.17g,%.17g,%.17g) actual=(%.17g,%.17g,%.17g) timing_ratio=%.17g\n",
                            i,kick0[i,:]...,d[i,:]...,ratio[i])
                end
                @assert ratio_resid<2e-7
                @assert delta_ratio_resid<1e-9
                if dx!=0 || dy!=0
                    @assert rel>0.02
                end
            else
                @assert count(==(BT.STATE_ALIVE),got.state)==0
                println("LOSS states=",got.state)
                # The same false losses occur with only 16 physical particles.
                low=tracked(w,K,true,simd;weight=1.0)
                @assert all(==(BT.STATE_LOST),low.state)
                println("LOW_POPULATION expected_alive=16 actual_alive=",count(==(BT.STATE_ALIVE),low.state))
            end
        end
        # A zero solenoid removes the canonical/mechanical difference. IBS still
        # runs and gives identical momenta after an ordinary spatial translation.
        vfree=initial(0.0)
        a=tracked(vfree,0.0,true,simd)
        w=translate_beam(vfree,0.75,0.25,0.0)
        b=tracked(w,0.0,true,simd)
        err=maximum(abs,b.v-translate_beam(a.v,0.75,0.25,0.0))
        @assert all(==(BT.STATE_ALIVE),b.state) && err<2e-15
        println("CONTROL K=0 translation_error=",err," alive=16")
    end
    println("DiscoveryChecks PASS")
    end

    println("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

