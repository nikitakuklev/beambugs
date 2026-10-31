---
title: "OPT-32: Montague functions omit momentum normalization"
hide:
  - toc
---

# OPT-32 — Montague functions omit momentum normalization

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on SciBmad main (`ae9423d`) |
| Source | [`src/twiss/twiss_operators.jl:849-911`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L849-L911)<br>[`src/twiss/twiss.jl:175-180`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L175-L180)<br>[`bmad/doc/optics.tex:505-527`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/doc/optics.tex#L505-L527) |

In uncoupled transverse optics, the scalar Montague chromatic functions `w`, W = √(A² + B²) built from the δ-derivatives β′ and α′ of the Twiss functions, use canonical-coordinate beta derivatives without the standard momentum normalization. With δ the relative momentum offset, the angle-coordinate beta is β_angle = (1+δ)β_c, so B requires β_c′/β_c + 1/(1+δ), and A = α_c′ − α_cB. Analytic drift transport and exact rational propagation confirm that momentum-independent angle ellipses have W = 0. For entrance (β_angle, α) = (2 m, 0.3) and (3 m, −0.2), `twiss` instead gives W = 1.044 and 1.020 initially, then 1.080 and 1.193 after 1.3 m of drift. This corrupts the chromatic-amplitude diagnostic without invalidating the canonical-coordinate beta convention itself.

??? example "Reproducer: OPT-32.jl"

    [Download OPT-32.jl](OPT-32.jl)

    ```julia
    # OPT-32: Montague functions omit momentum normalization
    #
    # In uncoupled transverse optics, the scalar Montague chromatic functions `w`, W = √(A² + B²) built
    # from the δ-derivatives β′ and α′ of the Twiss functions, use canonical-coordinate beta derivatives
    # without the standard momentum normalization. With δ the relative momentum offset, the
    # angle-coordinate beta is β_angle = (1+δ)β_c, so B requires β_c′/β_c + 1/(1+δ), and A = α_c′ −
    # α_cB. Analytic drift transport and exact rational propagation confirm that momentum-independent
    # angle ellipses have W = 0. For entrance (β_angle, α) = (2 m, 0.3) and (3 m, −0.2), `twiss` instead
    # gives W = 1.044 and 1.020 initially, then 1.080 and 1.193 after 1.3 m of drift. This corrupts the
    # chromatic-amplitude diagnostic without invalidating the canonical-coordinate beta convention
    # itself.
    #
    # The script supplies a canonical initial map representing momentum-independent angle-coordinate
    # ellipses with these entrance values, including the longitudinal canonical completion of the
    # transverse dilation, and propagates it on a uniform `Descriptor(6, 3)` through 0.4 m and 0.9 m
    # drifts. At each point it compares beta, alpha and their δ-derivatives with analytic drift
    # transport, and the reported W with the normalized W, which must be zero. Exact rational
    # propagation at δ = ±1/1024 independently shows that the angle ellipse does not depend on momentum.
    using SciBmad, LinearAlgebra, Printf
    BLAS.set_num_threads(1)
    function main()
        x,px,y,py,z,delta = vars(Descriptor(6,3))
        b1,a1 = 2.,.3
        b2,a2 = 3.,-.2
        u1,w1 = sqrt(b1)*x, (px-a1*x)/sqrt(b1)
        u2,w2 = sqrt(b2)*y, (py-a2*y)/sqrt(b2)
        q = 1+delta
        # Canonical completion of the momentum-dependent transverse dilation.
        initial = DAMap(nv=5,v=[u1/sqrt(q),w1*sqrt(q),u2/sqrt(q),w2*sqrt(q),z-(u1*w1+u2*w2)/(2q),delta])
        line = Beamline([Drift(L=.4),Drift(L=.9)],species_ref=Species("electron"),pc_ref=1e9)
        tw = twiss(line;a_initial=initial,order=3,chrom=3,cols=["beta1","alpha1","dbeta1","dalpha1","w1a","w1b","w1","beta2","alpha2","dbeta2","dalpha2","w2"])
        for (mode,beta0,alpha0) in ((1,b1,a1),(2,b2,a2))
            beta = getproperty(tw,Symbol("beta",mode))
            alpha = getproperty(tw,Symbol("alpha",mode))
            dbeta = getproperty(tw,Symbol("dbeta",mode))
            dalpha = getproperty(tw,Symbol("dalpha",mode))
            reported = getproperty(tw,Symbol("w",mode))
            for i in eachindex(tw.s)
                s = tw.s[i]
                # Analytic drift: β_c(s) = (β0 − 2α0 s + γ0 s²)/q and α_c(s) = α0 − γ0 s, so at δ = 0
                # the δ-derivatives are β′/β = −1 and α′ = 0, and the unnormalized W is √(1 + α²).
                gamma0 = (1+alpha0^2)/beta0
                expected_beta = beta0-2alpha0*s+gamma0*s*s
                expected_alpha = alpha0-gamma0*s
                # Energy-normalized Montague components at δ = 0: B = β′/β + 1, from β_angle = (1+δ)
                # β_c, and A = α′ − α B.
                normalized_b = dbeta[i]/beta[i]+1
                normalized_a = dalpha[i]-alpha[i]*normalized_b
                @printf("mode=%d s=%.3g beta=%.16g alpha=%.16g dbeta_over_beta=%.16g dalpha=%.3g reported_W=%.16g normalized_W=%.3g\n",mode,s,beta[i],alpha[i],dbeta[i]/beta[i],dalpha[i],reported[i],hypot(normalized_a,normalized_b))
                @assert abs(beta[i]-expected_beta) < 4e-14
                @assert abs(alpha[i]-expected_alpha) < 4e-14
                @assert abs(dbeta[i]+expected_beta) < 4e-14
                @assert abs(dalpha[i]) < 4e-14
                @assert abs(reported[i]-hypot(1,expected_alpha)) < 4e-14
                @assert hypot(normalized_a,normalized_b) < 4e-14
            end
            # Independent exact-rational propagation of the physical angle ellipse.
            # Start with Sigma_c(delta) = [beta/q -alpha; -alpha q*gamma].
            # Angle emittance is epsilon_c/q, so Sigma_angle/epsilon_angle
            # is q*P*Sigma_c*P', P=diag(1,1/q).
            br,ar = mode == 1 ? (2//1,3//10) : (3//1,-1//5)
            C = [br -ar; -ar (1+ar^2)/br]
            for s in (0//1, 13//10), delta in (-1//1024, 1//1024)
                q = 1+delta
                Ccanonical = [br/q -ar; -ar q*(1+ar^2)/br]
                Dcanonical = [1//1 s/q; 0//1 1//1]
                P = [1//1 0//1; 0//1 1/q]
                Dangle = [1//1 s; 0//1 1//1]
                actual = q*P*Dcanonical*Ccanonical*Dcanonical'*P'
                @assert actual == Dangle*C*Dangle'
            end
            println("mode=",mode," independent_exact_rational_angle_ellipse_delta_invariance=true")
        end
        println("DEFECT REPRODUCED")
    end
    main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | SciBmad main (`5f0fb83ac889`), pinned clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | SciBmad main (`5f0fb83ac889`), pinned clones, scibmad-ref environment, Julia 1.13.2, CPU, Float64, serial |

