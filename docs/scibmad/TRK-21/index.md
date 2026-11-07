---
title: "TRK-21: Stored zero dipole strength corrupts coordinates at bend faces"
hide:
  - toc
---

# TRK-21 — Stored zero dipole strength corrupts coordinates at bend faces

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [ext/BeamTrackingBeamlinesExt/unpack_bl.jl:215–242](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L215-L242)<br>[Beamlines.jl/src/element.jl:12](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/element.jl#L12)<br>[ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl:25](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/scibmadstandard_bl.jl#L25)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:415–444](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L415-L444)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:2–17](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L2-L17)<br>[src/kernels/integrators.jl:5–31](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/integrators.jl#L5-L31)<br>[src/kernels/fringe.jl:13–18](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/fringe.jl#L13-L18)<br>[src/kernels/fringe.jl:62–68](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/fringe.jl#L62-L68)<br>[src/kernels/fringe.jl:300](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/fringe.jl#L300)<br>[src/kernels/fringe.jl:311–312](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/fringe.jl#L311-L312)<br>[Beamlines.jl/src/bend.jl:29–34](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/bend.jl#L29-L34)<br>[docs/src/coordinates.md:160–179](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/coordinates.md#L160-L179)<br>[docs/src/coordinates.md:558–600](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/coordinates.md#L558-L600)<br>[ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:398–412](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L398-L412)<br>[src/kernels/bend_kick.jl:129–148](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/bend_kick.jl#L129-L148) |

Storing an explicit zero dipole strength in a bend with an enabled nonzero face angle produces NaN coordinates and can mark particles lost. With zero field each particle must follow a straight line to the reference exit plane, as an independent high-precision geometry calculation confirms. Across four geometries with lengths 0.3000–1.100 m and curvatures −0.2500 to 0.4000 m⁻¹, all 24 default-method cases lose all 16 particles instead of transporting them with finite coordinates. Omitting the field entry, setting both face angles to zero, disabling fringes or selecting `Exact()` preserves every particle; in mixed-strength batches exactly the zero-strength lanes fail.

Run `julia TRK-21.jl`.

??? example "Reproducer: TRK-21.jl"

    [Download TRK-21.jl](TRK-21.jl)

    ```julia
    # TRK-21: Stored zero dipole strength corrupts coordinates at bend faces
    #
    # Storing an explicit zero dipole strength in a bend with an enabled nonzero face angle produces NaN
    # coordinates and can mark particles lost. With zero field each particle must follow a straight line
    # to the reference exit plane, as an independent high-precision geometry calculation confirms.
    # Across four geometries with lengths 0.3000–1.100 m and curvatures −0.2500 to 0.4000 m⁻¹, all 24
    # default-method cases lose all 16 particles instead of transporting them with finite coordinates.
    # Omitting the field entry, setting both face angles to zero, disabling fringes or selecting
    # `Exact()` preserves every particle; in mixed-strength batches exactly the zero-strength lanes
    # fail.
    #
    # The script tracks 16 particles through bends of four geometries (lengths 0.3–1.1 m; curvatures 0,
    # −0.25, 0.4 and 0.22 m⁻¹; proton, electron and antiproton; one with a 0.31 rad reference roll) with
    # a stored zero dipole strength and nonzero face angles, with and without SIMD. It compares the
    # result with a 256-bit straight-ray calculation that calls no tracking kernel; omitting the field,
    # zeroing the face angles, disabling fringes and `Exact()` are controls that must match it. Further
    # checks cover other methods, single-face fringe selectors, mixed-strength batches, and a separately
    # written 12-proton reproduction.

    using SciBmad, LinearAlgebra
    BLAS.set_num_threads(1)

    # Work in Cartesian unit directions, intersect with the exit plane, and advance
    # physical arrival time. High precision avoids sharing the kernel's cancellation.
    function ray(v, L, g, ratio, roll)
        setprecision(256) do
            x, px, y, py, z, delta = BigFloat.(v)
            len, curv, r, alpha = BigFloat.((L, g, ratio, roll))
            ca, sa = cos(alpha), sin(alpha)
            x, y = ca*x + sa*y, -sa*x + ca*y
            px, py = ca*px + sa*py, -sa*px + ca*py
            u = 1 + delta
            dx, dy, ds = px/u, py/u, sqrt(u^2-px^2-py^2)/u
            theta = curv*len
            c, s = cos(theta), sin(theta)
            normal_dot_d = -s*dx + c*ds
            normal_dot_ref = curv == 0 ? len : s/curv
            distance = (normal_dot_ref + s*x)/normal_dot_d
            @assert distance > 0 && normal_dot_d > 0
            reference_x = curv == 0 ? zero(curv) : -2sin(theta/2)^2/curv
            xo = c*x + distance*(c*dx+s*ds) + reference_x
            yo = y + distance*dy
            pxo = u*(c*dx+s*ds)
            pyo = py
            beta = u/sqrt(u^2 + inv(r)^2)
            beta0 = r/sqrt(1+r^2)
            zo = z + len*beta/beta0 - distance
            return Float64.([ca*xo-sa*yo, ca*pxo-sa*pyo, sa*xo+ca*yo,
                sa*pxo+ca*pyo, zo, delta]), Float64(distance), Float64(normal_dot_d)
        end
    end

    function track_case(v, sp, ratio, L, g, roll, e1, e2, field, tm, simd)
        kw = field == :absent ? (;) : field == :Bn0 ? (; Bn0=0.0) : (; Kn0=field)
        ele = LineElement(; L, g_ref=g, tilt_ref=roll, e1, e2, tracking_method=tm, kw...)
        pc = ratio*massof(sp)
        line = Beamline([ele]; species_ref=sp, pc_ref=pc)
        b = Bunch(copy(v); species=sp, p_over_q_ref=pc/(chargeof(sp)*C_LIGHT))
        track!(b, line; use_explicit_SIMD=simd)
        return b
    end

    function main()
        n = 16
        v = hcat(collect(range(-0.017,0.023;length=n)), collect(range(-0.07,0.09;length=n)),
            collect(range(0.018,-0.013;length=n)), collect(range(-0.024,0.031;length=n)),
            collect(range(-0.04,0.05;length=n)), collect(range(-0.25,0.3;length=n)))
        v[8,:] .= 0 # Also test the nominal ray.
        max_control_error = 0.0
        bad_cases = 0
        for (name, L, g, ratio, roll, species) in (
            ("straight_low_energy", 0.3, 0.0, 0.25, 0.0, "proton"),
            ("negative_curvature", 1.1, -0.25, 3.7, 0.0, "electron"),
            ("rolled_antiproton", 0.6, 0.4, 1.3, 0.31, "anti-proton"),
            ("positive_curvature", 0.45, 0.22, 0.8, 0.0, "proton"),
        )
            sp = Species(species)
            results = [ray(row,L,g,ratio,roll) for row in eachrow(v)]
            expected = reduce(vcat, transpose.(first.(results)))
            println("GEOMETRY ", name, " min_distance=", minimum(getindex.(results,2)),
                " min_forward_cos=", minimum(getindex.(results,3)))
            for simd in (false,true)
                for (label,e1,e2) in (("entrance",0.17,0.0),("exit",0.0,-0.08),("both",0.17,-0.08))
                    # For a nonzero enabled face angle the fringe calls exact_wedge!, whose angular
                    # expression divides by Kn0, so a stored zero strength puts Inf/NaN into y and z.
                    b = track_case(v,sp,ratio,L,g,roll,e1,e2,0.0,SciBmadStandard(),simd)
                    println("BAD ",name," ",label," simd=",simd," alive=",count(==(0x01),b.state),
                        " finite_rows=",count(row->all(isfinite,row),eachrow(b.v)),
                        " row1=",collect(b.v[1,:])," oracle1=",collect(expected[1,:]))
                    @assert all(row->any(x->!isfinite(x),row), eachrow(b.v))
                    @assert all(==(0x02), b.state)
                    bad_cases += 1
                end
                for (label,field,e1,e2,tm) in (
                    ("absent",:absent,0.17,-0.08,SciBmadStandard()),
                    ("zero_faces",0.0,0.0,0.0,SciBmadStandard()),
                    ("fringe_disabled",0.0,0.17,-0.08,Symplectic(order=2,fringe_at=Fringe.NoEnd)),
                    ("Exact",0.0,0.17,-0.08,Exact()),
                )
                    b = track_case(v,sp,ratio,L,g,roll,e1,e2,field,tm,simd)
                    err = maximum(abs,b.v-expected)
                    max_control_error = max(max_control_error,err)
                    println("CONTROL ",name," ",label," simd=",simd," max_error=",err,
                        " alive=",count(==(0x01),b.state))
                    @assert err < 2e-15 && all(==(0x01),b.state)
                end
            end
        end
        sp=Species("proton")
        for field in (-0.0,:Bn0)
            b=track_case(v,sp,0.8,0.45,0.22,0.0,0.17,-0.08,field,SciBmadStandard(),false)
            println("REPRESENTATION field=",field," alive=",count(==(0x01),b.state)," finite=",all(isfinite,b.v))
            @assert any(x->!isfinite(x),b.v)
        end
        expected=reduce(vcat,transpose.([ray(row,0.45,0.22,0.8,0.0)[1] for row in eachrow(v)]))
        for field in (1e-3,1e-4,1e-5,-1e-4)
            b=track_case(v,sp,0.8,0.45,0.22,0.0,0.17,-0.08,field,SciBmadStandard(),false)
            println("LIMIT K=",field," max_difference_from_ray=",maximum(abs,b.v-expected),
                " alive=",count(==(0x01),b.state))
            @assert all(isfinite,b.v) && all(==(0x01),b.state)
        end
        println("CONFIRMED bend bad_cases=",bad_cases," max_control_error=",max_control_error)
    end

    function branches()
        sp=Species("proton")
        n=16
        v=hcat(fill(0.013,n),fill(-0.027,n),fill(0.008,n),fill(0.019,n),zeros(n),
            collect(range(-0.1,0.2;length=n)))
        expected=reduce(vcat,transpose.([ray(row,0.5,-0.3,1.1,0.0)[1] for row in eachrow(v)]))
        for (label,tm) in (
            ("symplectic_order4",Symplectic(order=4,n_steps=3)),
            ("bendkick",BendKick(order=2,n_steps=2)),
            ("entrance_only",Symplectic(order=2,fringe_at=Fringe.EntranceEnd)),
            ("exit_only",Symplectic(order=2,fringe_at=Fringe.ExitEnd)),
        ), simd in (false,true)
            b=track_case(v,sp,1.1,0.5,-0.3,0.0,0.12,-0.09,0.,tm,simd)
            println("BRANCH ",label," simd=",simd," alive=",count(==(0x01),b.state),
                " finite_rows=",count(row->all(isfinite,row),eachrow(b.v)))
            @assert any(x->!isfinite(x),b.v)
        end
        for (label,tm,e1,e2) in (
            ("disabled_entrance",Symplectic(order=2,fringe_at=Fringe.ExitEnd),0.12,0.0),
            ("disabled_exit",Symplectic(order=2,fringe_at=Fringe.EntranceEnd),0.0,-0.09),
        )
            b=track_case(v,sp,1.1,0.5,-0.3,0.0,e1,e2,0.,tm,false)
            err=maximum(abs,b.v-expected)
            println("SELECTOR_CONTROL ",label," max_error=",err," alive=",count(==(0x01),b.state))
            @assert err<2e-15 && all(==(0x01),b.state)
        end
        ks=[isodd(i) ? 0.0 : (-1.0)^(i÷2)*1e-3 for i in 1:n]
        for simd in (false,true)
            b=track_case(v,sp,1.1,0.5,-0.3,0.0,0.12,-0.09,BatchParam(ks),SciBmadStandard(),simd)
            println("BATCH_STRENGTH simd=",simd," state=",b.state," finite_rows=",
                [all(isfinite,row) for row in eachrow(b.v)])
            @assert all(!=(0x01),b.state[1:2:end]) && all(==(0x01),b.state[2:2:end])
            @assert all(isfinite,b.v[2:2:end,:])
        end
        println("CONFIRMED selected-face and mixed-strength branches")
    end

    main()
    branches()

    module DiscoveryRayCheck
    using SciBmad, LinearAlgebra, Printf
    BLAS.set_num_threads(1)
    const sp=Species("proton")
    const pc=2massof(sp)
    const L=0.7
    const g=0.2
    const v0=hcat([0.01+0.001j for j=1:12], [0.008+0.002j for j=1:12],
        [-0.015+0.001j for j=1:12], [-0.009+0.001j for j=1:12], fill(0.02,12), [0.01j for j=1:12])

    # Cartesian straight ray intersected with the curved reference's exit plane.
    # tau is physical flight distance divided by the normalized total momentum.
    function straight_ray_oracle(v; curvature=g)
        theta=curvature*L
        c,s=cos(theta),sin(theta)
        rot=[c 0 s; 0 1 0; -s 0 c]
        origin=curvature==0 ? [0.0,0.0,L] : [(c-1)/curvature,0.0,s/curvature]
        u=1+v[6]
        p=[v[2],v[4],sqrt(u^2-v[2]^2-v[4]^2)]
        r=[v[1],v[3],0.0]
        rp=rot*(r-origin)
        pp=rot*p
        tau=-rp[3]/pp[3]
        @assert tau>0 && pp[3]>0
        out=rp+tau*pp
        beta=u/sqrt(u^2+(massof(sp)/pc)^2)
        beta0=pc/hypot(pc,massof(sp))
        return [out[1],pp[1],out[2],pp[2],v[5]-u*tau+L*beta/beta0,v[6]]
    end
    expected=reduce(vcat,transpose.(straight_ray_oracle.(eachrow(v0))))
    function tracked(;field=0.0, e1=0.1,e2=0.1,simd=false,
                      tm=SciBmadStandard(),curvature=g)
        strength=isnothing(field) ? (;) : (;Kn0=field)
        ele=LineElement(;L,g_ref=curvature,e1,e2,tracking_method=tm,strength...)
        b=Bunch(copy(v0);species=sp,p_over_q_ref=pc/C_LIGHT)
        track!(b,Beamline([ele];species_ref=sp,pc_ref=pc);use_explicit_SIMD=simd)
        return b
    end
    for simd in (false,true)
        b=tracked(;simd)
        println("DEFAULT simd=",simd," alive=",count(==(0x01),b.state)," finite=",all(isfinite,b.v))
        @assert any(!isfinite,b.v) && any(!=(0x01),b.state)
        if !simd
            for i=1:12
                println("particle=",i," initial=",v0[i,:]," expected=",expected[i,:]," actual=",b.v[i,:]," state=",b.state[i])
            end
        end
        for (label,kwargs) in (
            ("omitted_field",(;field=nothing)),
            ("zero_faces",(;e1=0.0,e2=0.0)),
            ("no_fringe",(;tm=Symplectic(order=2,fringe_at=Fringe.NoEnd))),
            ("Exact_field_free",(;tm=Exact())))
            control=tracked(;simd,kwargs...)
            err=maximum(abs,control.v-expected)
            println("CONTROL ",label," simd=",simd," max_error=",err," alive=",count(==(0x01),control.state))
            @assert err<2e-15 && all(==(0x01),control.state)
        end
    end
    for (e1,e2) in ((0.1,0.0),(0.0,0.1))
        b=tracked(;e1,e2)
        println("SINGLE_EDGE e1=",e1," e2=",e2," alive=",count(==(0x01),b.state)," finite=",all(isfinite,b.v)," row1=",b.v[1,:])
        @assert any(!isfinite,b.v)
    end
    for field in (1e-4,1e-8,1e-12)
        b=tracked(;field)
        println("WEAK_FIELD K=",field," max_difference_from_zero_field=",maximum(abs,b.v-expected)," alive=",count(==(0x01),b.state))
    end
    println("ZERO-FIELD BEND FRINGE DEFECT REPRODUCED")
    end

    println("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

