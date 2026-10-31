---
title: "OPT-35: GTPSA inversion loses parameter-dependent orbital coefficients"
hide:
  - toc
---

# OPT-35 — GTPSA inversion loses parameter-dependent orbital coefficients

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on GTPSA main (`a5790a7`) |
| Source | [`src/methods.jl:562-565`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/methods.jl#L562-L565)<br>[`src/methods.jl:532-544`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/methods.jl#L532-L544)<br>[`methods.jl:519-530`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/methods.jl#L519-L530)<br>[`:551-560`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/methods.jl#L551-L560)<br>[`src/low_level/rtpsa.jl:2128-2139`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/low_level/rtpsa.jl#L2128-L2139) |

GTPSA's allocating `inv`, and `inv!` into a zero destination, lose parameter-dependent orbital terms of nonlinear total degree, because the destination's parameter rows are not initialized to identity substitutions. The failure includes transfer maps linear in phase space and linear in a parameter. For X = 2x + 0.3p and P = −0.7x + 0.395p + 0.7Xab, with parameters a and b, the inverse's p coefficient of Xab is 0 instead of −1.400; the parameterized shear (x, p + κx²) likewise loses its inverse x²κ coefficient −1. Algebraic reversal, identity-seeded destinations and NonlinearNormalForm's independent inverse agree. Parameter-free and parameter-only linear controls pass; the impact on standard SciBmad optics was not established.

??? example "Reproducer: OPT-35.jl"

    [Download OPT-35.jl](OPT-35.jl)

    ```julia
    # OPT-35: GTPSA inversion loses parameter-dependent orbital coefficients
    #
    # GTPSA's allocating `inv`, and `inv!` into a zero destination, lose parameter-dependent orbital
    # terms of nonlinear total degree, because the destination's parameter rows are not initialized to
    # identity substitutions. The failure includes transfer maps linear in phase space and linear in a
    # parameter. For X = 2x + 0.3p and P = −0.7x + 0.395p + 0.7Xab, with parameters a and b, the
    # inverse's p coefficient of Xab is 0 instead of −1.400; the parameterized shear (x, p + κx²)
    # likewise loses its inverse x²κ coefficient −1. Algebraic reversal, identity-seeded destinations
    # and NonlinearNormalForm's independent inverse agree. Parameter-free and parameter-only linear
    # controls pass; the impact on standard SciBmad optics was not established.
    #
    # The script inverts the canonical map X = 2x + 0.3p, P = −0.7x + 0.395p + h(X, a, b), with identity
    # parameter rows, for h = cXa, cX²a, cXab, ca², ca and cX², coefficients c = 0.7, −1.3 and 0.4 +
    # 0.9i, and total orders 4 and 6. It compares each inverse with the algebraic reversal of the linear
    # map and shear, with `inv!` into a destination seeded with identity parameter rows, and with
    # NonlinearNormalForm's `DAMap` inverse, and scales the seeded parameter rows to show how the result
    # depends on them. A parameter-free map and the shear (x, p + cκx², κ) at orders 3 and 5 complete
    # the checks.
    using SciBmad, LinearAlgebra, Printf
    BLAS.set_num_threads(1)
    const G=GTPSA
    const N=NonlinearNormalForm
    err(a,b)=maximum(G.normTPS.(a.-b))

    function main()
        # Compose a canonical linear map A with a parameter-dependent shear.
        # A=[2 .3; -.7 .395], det(A)=1. The inverse follows by reversing
        # these two explicit operations; it requires no numerical inversion.
        for order in (4,6), c in (0.7,-1.3,0.4+0.9im)
            d=Descriptor(2,order,2,order)
            xr,pr=vars(d); ar,br=params(d)
            unit=one(c)
            x,p,a,b=unit*xr,unit*pr,unit*ar,unit*br
            id=[x,p,a,b]
            for (label,n,u,v) in (("x_a",1,1,0),("x2_a",2,1,0),
                                   ("x_a_b",1,1,1),("a2",0,2,0),
                                   ("linear_a",0,1,0),("x2",2,0,0))
                h(q)=c*q^n*a^u*b^v
                X=2x+0.3p
                Y=-0.7x+0.395p+h(X)
                f=[X,Y,a,b]
                pi=p-h(x)
                expected=[0.395x-0.3pi,0.7x+2pi,a,b]
                result=inv(f)
                zero_dest=zero.(f)
                G.inv!(zero_dest,f)
                seeded=copy.(id)
                G.inv!(seeded,f)
                # Destination orbital garbage should always be overwritten.
                garbage=[x+x^2,p-p^3,a,b]
                G.inv!(garbage,f)
                nnf_result=inv(N.DAMap(nv=2,v=copy.(f[1:2])))
                eo=err(result[1:2],expected[1:2])
                es=err(seeded,expected)
                en=err(nnf_result.v,expected)
                eg=err(garbage,expected)
                @printf("CASE order=%d c=%s monomial=%s orbital_error=%.17g seeded=%.17g nnf=%.17g orbital_garbage=%.17g zero_dest_equivalence=%.17g\n",order,string(c),label,eo,es,en,eg,err(zero_dest,result))
                @assert es<2e-13 && en<2e-13 && eg<2e-13
                @assert err(zero_dest,result)<2e-13
                if u+v>0 && n+u+v>1
                    @assert eo>0.1
                else
                    @assert eo<2e-13
                end
                @assert G.normTPS(result[3])==0 && G.normTPS(result[4])==0
                if order==4 && c==0.7 && label=="x_a_b"
                    # Keep external parameters fixed when evaluating orbital rows.
                    # This removes returned parameter metadata from the oracle.
                    for q in ([0.08,-0.03,0.4,-0.6],[-0.06,0.02,-0.2,0.3])
                        Xq=2q[1]+0.3q[2]
                        Yq=-0.7q[1]+0.395q[2]+c*Xq*q[3]*q[4]
                        forward=[Xq,Yq,q[3],q[4]]
                        actual=G.evaluate.(result[1:2],Ref(forward))
                        exact=G.evaluate.(expected[1:2],Ref(forward))
                        @printf("POINT q=%s actual=%s exact=%s residual=%s\n",string(q),string(actual),string(exact),string(actual-q[1:2]))
                        @assert maximum(abs,exact-q[1:2])<2e-16
                    end
                    # Probe destination-parameter dependence without changing f.
                    for scale in (0.0,0.5,1.0,2.0)
                        dest=[zero(x),zero(p),scale*a,scale*b]
                        G.inv!(dest,f)
                        @printf("SEED scale=%.6g inverse_p[x*a*b]=%s expected=%s\n",scale,string(dest[2][[1,0,1,1]]),string(-2c))
                        @assert abs(dest[2][[1,0,1,1]]+2c*scale^2)<2e-14
                    end
                end
            end
        end
        # A descriptor without parameters provides a separate dimensional control.
        d0=Descriptor(2,4); x0,p0=vars(d0)
        f0=[x0,p0-0.9x0^3]
        e0=err(inv(f0),[x0,p0+0.9x0^3])
        @printf("NO_PARAMETERS coefficient_error=%.17g\n",e0)
        @assert e0<1e-14
        # The discovery's simple shear gives an exact-coefficient control at the
        # minimal total order that retains x^2*k, with real and complex coefficients.
        for order in (3,5), c in (1.0,1.0+0.5im)
            d=Descriptor(2,order,1,order)
            xr,pr=vars(d); kr=only(params(d))
            x,p,k=one(c)*xr,one(c)*pr,one(c)*kr
            f=[x,p+c*k*x^2,k]
            expected=[x,p-c*k*x^2,k]
            result=inv(f)
            seeded=[copy(x),copy(p),copy(k)]
            G.inv!(seeded,f)
            nnf=inv(N.DAMap(nv=2,v=copy.(f[1:2])))
            @printf("SHEAR order=%d c=%s actual_p[x2*k]=%s expected=%s seeded=%.17g nnf=%.17g\n",order,string(c),string(result[2][[2,0,1]]),string(-c),err(seeded,expected),err(nnf.v,expected))
            @assert result[2][[2,0,1]]==0
            @assert err(seeded,expected)==0 && err(nnf.v,expected)==0
        end
        println("DEFECT REPRODUCED")
    end
    main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | GTPSA 1.5.6 (`a5790a7deaa6`), pinned clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | GTPSA 1.5.6 (`a5790a7deaa6`), pinned clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64, serial |

