---
title: "OPT-36: Cross-descriptor TPS replacement retains destination coefficients"
hide:
  - toc
---

# OPT-36 — Cross-descriptor TPS replacement retains destination coefficients

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on GTPSA release [v1.5.6](https://github.com/bmad-sim/GTPSA.jl/tree/v1.5.6) (`a5790a7`) |
| Source | [`src/methods.jl:39-49`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/methods.jl#L39-L49)<br>[`src/low.jl:8-27`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/low.jl#L8-L27)<br>[`src/low_level/rtpsa.jl:477-495`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/low_level/rtpsa.jl#L477-L495)<br>[`:613-625`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/low_level/rtpsa.jl#L613-L625)<br>[`TPSAInterface.jl/ext/TPSAInterfaceGTPSAExt.jl:66`](https://github.com/bmad-sim/TPSAInterface.jl/blob/59800445670c542ac4546a6562af31ca3c3335a9/ext/TPSAInterfaceGTPSAExt.jl#L66)<br>[`NonlinearNormalForm.jl/src/methods.jl:31-44`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/methods.jl#L31-L44)<br>[`src/sanity.jl:20-29`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/sanity.jl#L20-L29)<br>[`GTPSA.jl/src/methods.jl:17-19`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/methods.jl#L17-L19)<br>[`TPSAInterface.jl/src/TPSAInterface.jl:102-105`](https://github.com/bmad-sim/TPSAInterface.jl/blob/59800445670c542ac4546a6562af31ca3c3335a9/src/TPSAInterface.jl#L102-L105) |

Copying into a destination with a different descriptor through GTPSA's `setTPS!(…; change=true)`, `copy_tps!` or orbital map `copy!` keeps old destination coefficients that are absent from the source. Whole-series replacement must instead equal the source projected onto the destination's admitted monomials, as independent coefficient enumeration and fresh-storage controls establish. Copying from total/parameter orders (6, 2) to (3, 1) leaves a stale constant 5.000 and a coefficient 1-norm error of 33.00. Copying a rotation over storage that held a sextupole shear gives a local Jacobian determinant of 0.6500 at x = 0.125 instead of 1, violating symplecticity. Explicit clearing, fresh destinations and same-descriptor copies pass.

Run `julia OPT-36.jl`.

??? example "Reproducer: OPT-36.jl"

    [Download OPT-36.jl](OPT-36.jl)

    ```julia
    # OPT-36: Cross-descriptor TPS replacement retains destination coefficients
    #
    # Copying into a destination with a different descriptor through GTPSA's `setTPS!(…; change=true)`,
    # `copy_tps!` or orbital map `copy!` keeps old destination coefficients that are absent from the
    # source. Whole-series replacement must instead equal the source projected onto the destination's
    # admitted monomials, as independent coefficient enumeration and fresh-storage controls establish.
    # Copying from total/parameter orders (6, 2) to (3, 1) leaves a stale constant 5.000 and a
    # coefficient 1-norm error of 33.00. Copying a rotation over storage that held a sextupole shear
    # gives a local Jacobian determinant of 0.6500 at x = 0.125 instead of 1, violating symplecticity.
    # Explicit clearing, fresh destinations and same-descriptor copies pass.
    #
    # The script copies the admitted part of 2x − 3pa + 0.25x²p − 4a² + 6x⁴, with parameter a, into a
    # destination holding 5 + 11p + 13a² + 17x³ + 19x²p + 23x⁵, in both cap directions, for real and
    # complex coefficients, dynamic and static descriptors and both copy routines, and compares the
    # result with a dense enumeration of every admitted monomial; it also copies a zero series, and
    # scalar-zero assignment, clearing before the copy, a fresh destination and a same-descriptor copy
    # are controls. `DAMap` and `TPSAMap` copies of a rotation into a sextupole shear's storage are
    # checked against the analytic Jacobian determinant. A separate, independently written module checks
    # order-5→4 and order-3→5 polynomials and a drift copied over a lens map.

    module IndependentChecks
    # Independent whole-polynomial oracle: enumerate every destination-admissible
    # monomial, including zero source coefficients. No source iterator is used.
    using SciBmad, LinearAlgebra, Printf
    BLAS.set_num_threads(1)
    const G = SciBmad.GTPSA
    const TI = SciBmad.TI
    const N = SciBmad.NNF

    allowed(m, order, po) = sum(m) <= order && m[3] <= po
    monos(order, po) = [(i,j,k) for i in 0:order for j in 0:order for k in 0:po if i+j+k <= order]
    function polynomial(d, T, terms, order, po; static=false)
        t = static ? G.TPS{T,d}() : G.TPS{T,G.Dynamic}(use=d)
        scale = T === Float64 ? 1.0 : 1.0 + 0.25im
        for (m,c) in terms
            allowed(m,order,po) && (t[collect(m)] = scale*c)
        end
        return t
    end
    function dense_error(t, terms, order, po, T)
        scale = T === Float64 ? 1.0 : 1.0 + 0.25im
        return sum(abs(t[collect(m)] - scale*get(terms,m,0)) for m in monos(order,po))
    end

    source_terms = Dict((1,0,0)=>2, (0,1,1)=>-3, (2,1,0)=>0.25,
        (0,0,2)=>-4, (4,0,0)=>6)
    old_terms = Dict((0,0,0)=>5, (0,1,0)=>11, (0,0,2)=>13,
        (3,0,0)=>17, (2,1,0)=>19, (5,0,0)=>23)
    for (so,sp,do_,dp) in ((6,2,3,1), (3,1,6,2)), T in (Float64,ComplexF64), static in (false,true)
        ds = Descriptor(2,so,1,sp)
        dd = Descriptor(2,do_,1,dp)
        @assert ds != dd
        src = polynomial(ds,T,source_terms,so,sp;static)
        # Source coefficients outside its descriptor are identically absent.
        expected = Dict(m=>c for (m,c) in source_terms if allowed(m,so,sp) && allowed(m,do_,dp))
        errors = Float64[]
        for api in (:setTPS, :interface)
            dest = polynomial(dd,T,old_terms,do_,dp;static)
            if api === :setTPS
                G.setTPS!(dest,src;change=true)
            else
                TI.copy_tps!(dest,src)
            end
            e = dense_error(dest,expected,do_,dp,T)
            push!(errors,e)
            @assert e > 30
            @assert dest[0] == 5*(T===Float64 ? 1 : 1+0.25im)
            @assert dest[[2,1,0]] == 0.25*(T===Float64 ? 1 : 1+0.25im)
            println("PROJECTION api=",api," type=",T," static=",static,
                " orders=",(so,sp,do_,dp)," coefficient_l1_error=",e,
                " stale_constant=",dest[0]," shared_x2p=",dest[[2,1,0]])
        end
        @assert errors[1] == errors[2]
        zero_source = zero(src)
        dest = polynomial(dd,T,old_terms,do_,dp;static)
        old_norm = G.normTPS(dest)
        TI.copy_tps!(dest,zero_source)
        @assert G.normTPS(dest) == old_norm
        # Copying scalar zero follows the clearing branch.
        TI.copy_tps!(dest,0)
        @assert G.normTPS(dest) == 0
        dest = polynomial(dd,T,old_terms,do_,dp;static)
        G.clear!(dest)
        TI.copy_tps!(dest,src)
        @assert dense_error(dest,expected,do_,dp,T) == 0
        # Allocating conversion starts with empty storage.
        converted = G.TPS{T,G.Dynamic}(src;use=dd)
        @assert dense_error(converted,expected,do_,dp,T) == 0
        # Same descriptor uses native whole-series copy.
        same = polynomial(ds,T,old_terms,so,sp;static)
        TI.copy_tps!(same,src)
        source_expected = Dict(m=>c for (m,c) in source_terms if allowed(m,so,sp))
        @assert dense_error(same,source_expected,so,sp,T) == 0
        println("CONTROLS zero_TPS_retained_norm=",old_norm,
            " scalar_zero_norm=0 clear_then_copy_error=0 allocating_error=0 same_descriptor_error=0")
    end

    # A nonlinear destination term tests area preservation away from the origin.
    # Hamiltonian rotation R=(c*x+s*p,-s*x+c*p), c=3/5,s=4/5; an old thin
    # sextupole shear leaves k*x^2 in R's second component. Both linear terms
    # are present in the source, so its determinant becomes 1-2*k*s*x.
    for Map in (N.DAMap,N.TPSAMap)
        ds = Descriptor(2,4)
        xs,ps = vars(ds)
        dd = Descriptor(2,6)
        x,p = vars(dd)
        k = 1.75
        c,s = 3/5,4/5
        source = Map(nv=2,v=[c*xs+s*ps,-s*xs+c*ps])
        dest = Map(nv=2,v=[x,p+k*x^2])
        @assert source.q === nothing && dest.q === nothing
        copy!(dest,source)
        J = G.jacobian(dest.v)
        @assert J == [c s; -s c]
        # Jacobian at zero passes despite the wrong nonlinear map.
        for q in ([0.125,-0.0625],[-0.25,0.03125])
            got = G.evaluate(dest.v,q)
            exact = [c*q[1]+s*q[2],-s*q[1]+c*q[2]]
            jq = [G.evaluate(G.deriv(dest.v[i],j),q) for i in 1:2,j in 1:2]
            @assert maximum(abs,got-(exact+[0,k*q[1]^2])) < 2e-16
            @assert abs(det(jq)-(1-2*k*s*q[1])) < 3e-16
            println("MAP type=",Map," ray=",q," expected=",exact," actual=",got,
                " det_at_origin=",det(J)," expected_local_det=1 actual_local_det=",det(jq))
        end
        fresh = Map(nv=2,v=[zero(x),zero(p)])
        copy!(fresh,source)
        @assert G.normTPS(fresh.v[1]-(c*x+s*p))+G.normTPS(fresh.v[2]-(-s*x+c*p)) == 0
        println("MAP_CONTROL type=",Map," fresh_destination_error=0")
    end
    println("IndependentChecks PASS")
    end

    module DiscoveryChecks
    using SciBmad, LinearAlgebra, Printf
    const G=SciBmad.GTPSA
    const TI=SciBmad.TI
    const N=SciBmad.NNF
    BLAS.set_num_threads(1)

    for T in (Float64,ComplexF64), (mo1,mo2) in ((5,4),(3,5))
        d1=Descriptor(2,mo1); d2=Descriptor(2,mo2)
        x=G.vars(d1); y=G.vars(d2)
        c=T==Float64 ? 1.0 : 1.0+0.5im
        source=c*(1+2x[1]+3x[1]^2*x[2])
        expected=c*(1+2y[1]+3y[1]^2*y[2])
        old=c*(7+11y[2]+13y[1]^3)
        target=copy(old)
        G.setTPS!(target,source;change=true)
        residual=G.normTPS(target-expected)
        println("SET type=",T," orders=",(mo1,mo2)," expected_residual=0 actual_residual=",residual,
                " stale_y=",target[[0,1]]," stale_x3=",target[[3,0]])
        @assert G.normTPS(target-expected-c*(11y[2]+13y[1]^3))==0
        # The public generic copy route calls the same conversion.
        via_interface=copy(old); TI.copy_tps!(via_interface,source)
        @assert G.normTPS(via_interface-target)==0
        println("INTERFACE copy_matches_failure=true")
        G.setTPS!(target,zero(source);change=true)
        println("ZERO_SOURCE expected_norm=0 actual_norm=",G.normTPS(target))
        @assert G.normTPS(target)==G.normTPS(via_interface)
        # An explicit zero coefficient in a sparse source must still overwrite.
        clean=zero(old); G.setTPS!(clean,source;change=true)
        same=c*(7+11x[2]+13x[1]^3); G.setTPS!(same,source;change=true)
        numeric=copy(old); G.setTPS!(numeric,0;change=true)
        newcopy=G.TPS{T,G.Dynamic}(source;use=d2)
        @assert G.normTPS(clean-expected)==0
        @assert G.normTPS(same-source)==0
        @assert G.normTPS(numeric)==0
        @assert G.normTPS(newcopy-expected)==0
        println("CONTROLS zero_destination=0 same_descriptor=0 numeric_zero=0 allocating_conversion=0")
    end

    # A Hamiltonian free shear H=p^2/2 has q'=q+ell*p and p'=p,
    # so its Jacobian determinant and symplectic area factor are exactly one.
    # Reuse a former thin-lens map as the destination of the public NNF copy!.
    d1=Descriptor(2,3); d2=Descriptor(2,5)
    x=G.vars(d1); y=G.vars(d2)
    ell=0.75; k=0.5
    src=DAMap(nv=2,v=[x[1]+ell*x[2],x[2]])
    dst=DAMap(nv=2,v=[y[1],y[2]-k*y[1]])
    copy!(dst,src)
    J=G.jacobian(dst.v); S=[0.0 1.0;-1.0 0.0]
    println("MAP expected_jacobian=",[1.0 ell;0.0 1.0]," actual_jacobian=",J)
    println("MAP expected_det=1 actual_det=",det(J)," symplectic_residual=",maximum(abs,J'*S*J-S))
    @assert J==[1.0 ell;-k 1.0]
    @assert det(J)==1+k*ell
    for i=1:16
        q=(i-8.5)/1000; p=(i%5-2)/10000
        expected=[q+ell*p,p]
        actual=G.evaluate(dst.v,[q,p])
        @printf("RAY %d input=(%.17g,%.17g) expected=(%.17g,%.17g) actual=(%.17g,%.17g)\n",
                i,q,p,expected...,actual...)
        @assert abs(actual[2]-(p-k*q))<2e-18
    end
    fresh=DAMap(nv=2,v=zero.(y)); copy!(fresh,src)
    @assert G.jacobian(fresh.v)==[1.0 ell;0.0 1.0]
    println("MAP_CONTROL fresh_destination_det=",det(G.jacobian(fresh.v)))
    println("DiscoveryChecks PASS")
    end

    println("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | GTPSA 1.5.6 (`a5790a7deaa6`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | GTPSA 1.5.6 (`a5790a7deaa6`), source, Julia 1.13.2, CPU, Float64, serial |

