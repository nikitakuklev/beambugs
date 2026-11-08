---
title: "OPT-37: Repeated eigenvalues lose conjugate eigenvector pairing"
hide:
  - toc
---

# OPT-37 — Repeated eigenvalues lose conjugate eigenvector pairing

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on NonlinearNormalForm commit [`dfc5c26`](https://github.com/bmad-sim/NonlinearNormalForm.jl/commit/dfc5c2697e53), committed 2026-11-06 |
| Issue | [[gh:https://github.com/bmad-sim/NonlinearNormalForm.jl/issues/65]] |
| Source | [`mat_eigen`, `src/matrix.jl:2–26`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/matrix.jl#L2-L26)<br>[`mat_eigen!`, `src/matrix.jl:31–39`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/matrix.jl#L31-L39)<br>[`moveback_unstable!`, `src/matrix.jl:241–254`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/matrix.jl#L241-L254)<br>[`normalize_eigenmode!`, `src/matrix.jl:172–205`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/matrix.jl#L172-L205)<br>[`orthogonalize_degenerate_pairs!`, `src/matrix.jl:124–150`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/matrix.jl#L124-L150)<br>[`src/normal.jl:100–108`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L100-L108)<br>[`equilibrium_moments`, `src/normal.jl:547–589`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L547-L589)<br>[exported, `src/NonlinearNormalForm.jl:58–72`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/NonlinearNormalForm.jl#L58-L72)<br>[damped maps, `docs/src/index.md:9`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/docs/src/index.md#L9)<br>[normalization and equilibrium covariance, `docs/src/index.md:36–46`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/docs/src/index.md#L36-L46) |

With repeated eigenvalues, `mat_eigen` can pair eigenvectors from different modes using conjugate eigenvalues alone. An uncoupled stable matrix returns 12 nonfinite entries with default options; disabling mode sorting can instead return finite vectors with an 11.61% relative eigenvector residual. Default equilibrium covariance calculation also raises on a weakly damped map, although an independent high-precision solve gives a finite positive covariance. Preserving conjugate vector pairs removes these failures. Equal emittances with distinct transport tunes remain accurate.

Run `python OPT-37.py` with [_scibmad.py](../helpers/_scibmad.py) in the same directory.

??? example "Reproducer: OPT-37.py"

    [Download OPT-37.py](OPT-37.py)

    ```python
    """OPT-37: Repeated eigenvalues lose conjugate eigenvector pairing

    With repeated eigenvalues, `mat_eigen` can pair eigenvectors from different modes using conjugate eigenvalues alone. An uncoupled stable matrix returns 12 nonfinite entries with default options; disabling mode sorting can instead return finite vectors with an 11.61% relative eigenvector residual. Default equilibrium covariance calculation also raises on a weakly damped map, although an independent high-precision solve gives a finite positive covariance. Preserving conjugate vector pairs removes these failures. Equal emittances with distinct transport tunes remain accurate.

    Setup: `SCIBMAD_JULIA` set to a Julia launcher, an executable that starts Julia in the SciBmad environment; the Python script runs its embedded Julia program through it.

    The script constructs uncoupled canonical blocks, checks both eigensystem APIs, and compares equilibrium covariance with a 256-bit Kronecker solve and a doubled covariance sum. Analytic eigenvectors, preserved LAPACK pairs, a supplied normalizer and separated transport frequencies provide controls. All cases use fixed matrices and one numerical thread.
    """

    from _scibmad import run_julia

    _PROGRAM = r'''
    using SciBmad, LinearAlgebra, Printf
    const NNF = SciBmad.NonlinearNormalForm
    const BAD = Ref(false)

    function blocks(a, frequencies, betas)
        M = zeros(6, 6)
        for j in 1:3
            b, w = betas[j], frequencies[j]
            M[2j-1:2j, 2j-1:2j] = [a b*w; -w/b a]
        end
        M
    end
    basis(betas) = Matrix(Diagonal(reduce(vcat, ([sqrt(b), 1/sqrt(b)] for b in betas))))
    covariance(es, betas) = Matrix(Diagonal(reduce(vcat,
        ([es[j]*betas[j], es[j]/betas[j]] for j in 1:3))))

    function analytic_vectors(a, ws, betas)
        V, values = zeros(ComplexF64, 6, 6), zeros(ComplexF64, 6)
        for j in 1:3
            # Canonical J=[0 1;-1 0]: (sqrt(beta),i/sqrt(beta))/sqrt(2)
            # has symplectic norm +i and eigenvalue a+i*w in its own plane.
            V[2j-1, 2j-1], V[2j, 2j-1] = sqrt(betas[j]/2), im/sqrt(2betas[j])
            V[:, 2j] = conj(V[:, 2j-1])
            values[2j-1] = a + im*ws[j]
            values[2j] = conj(values[2j-1])
        end
        Eigen(values, V)
    end

    function metrics(label, M, F; control=false)
        nf = count(!isfinite, F.vectors)
        failed = nf != 0
        if nf == 0
            pair_error = maximum(norm(F.vectors[:,2j]-conj(F.vectors[:,2j-1])) for j in 1:3)
            residual = norm(M*F.vectors-F.vectors*Diagonal(F.values))/norm(M*F.vectors)
            dimension = rank(F.vectors)
            failed |= pair_error > 2e-14 || residual > 2e-14 || dimension != 6
            @printf("%s nonfinite=0 rank=%d expected_rank=6 pair_error=%.17g relative_eigen_residual=%.17g expected=0 atol=2e-14 derivation=analytic_block_eigenvectors\n",
                    label, dimension, pair_error, residual)
        else
            println(label, " nonfinite=", nf, " expected=0 derivation=analytic_finite_block_eigenvectors")
        end
        control && failed && error("control failed: $label")
        !control && (BAD[] |= failed)
    end

    function preserved_pairs(M; sort=true, phase=true)
        # Real LAPACK supplies adjacent conjugate columns. Preserve that order while
        # retaining the complete native normalization, including its orthogonalizer.
        F = eigen(Matrix(M); sortby=nothing)
        NNF.low_mat_eigen!(F, sort, phase, true)
    end

    function oracle(label, M, Q)
        setprecision(256) do
            mb, qb = BigFloat.(M), BigFloat.(Q)
            # Column-major vectorization gives vec(M Sigma M^T)=(M kron M)vec(Sigma).
            direct = reshape((Matrix{BigFloat}(I,36,36)-kron(mb,mb))\vec(qb), 6, 6)
            fp = norm(direct-mb*direct*transpose(mb)-qb)/norm(qb)
            # Repeated doubling evaluates 2^25 covariance-series terms without either
            # an eigensystem or the Kronecker solve, even for the weakly damped case.
            series, power = copy(qb), copy(mb)
            for _ in 1:25
                series += power*series*transpose(power)
                power *= power
            end
            disagreement = norm(series-direct)/norm(direct)
            @assert fp < big"1e-65" && disagreement < big"1e-65"
            @assert isposdef(Symmetric(direct))
            @printf("ORACLE %s bits=256 fixed_point_over_Q=%.17g series_vs_direct=%.17g expected=0 atol=1e-65 derivation=stored_binary64_inputs_Kronecker_and_2^25_term_sum\n",
                    label, Float64(fp), Float64(disagreement))
            Float64.(direct)
        end
    end

    function covariance_case(label, M, sigma, betas; control=false)
        Q = sigma-M*sigma*transpose(M)
        @assert isposdef(Symmetric(sigma)) && isposdef(Symmetric(Q))
        radius = maximum(abs, eigen(M).values)
        @assert radius < 1
        ref = oracle(label, M, Q)
        @printf("INPUT %s radius=%.17g min_noise_eigenvalue=%.17g analytic_vs_stored_oracle=%.17g expected=stable_positive_covariance\n",
                label, radius, eigmin(Symmetric(Q)), norm(sigma-ref)/norm(ref))
        ray = collect(vars(Descriptor(6,1)))
        m = DAMap(nv=6, v=ray, v_matrix=M, s=Q)
        known = DAMap(nv=6, v=ray, v_matrix=basis(betas), s=zeros(6,6))
        # Recreate normal(m,1)'s linear construction with pair-preserving left
        # eigenvectors. This leaves its later degenerate orthogonalizer unchanged.
        F = preserved_pairs(Matrix(transpose(M)); phase=false)
        Ainv = real(c_jacobian(m,NNF.HVARS)*transpose(F.vectors))
        paired = DAMap(nv=6, v=ray, v_matrix=inv(Ainv), s=zeros(6,6))
        # Weak damping amplifies binary64 rounding by 1/(1-|lambda|^2); its minimum
        # denominator is 1.9e-5. Compare with the stored-input oracle at this budget.
        tol = radius > .99 ? 5e-11 : 2e-14
        for (name, normalizer) in (("supplied",known), ("preserved_pairs",paired))
            result = equilibrium_moments(m,normalizer)
            err = norm(result-ref)/norm(ref)
            fp = norm(result-M*result*transpose(M)-Q)/norm(Q)
            @assert err < tol && fp < tol
            @printf("CONTROL %s %s relative_covariance_error=%.17g fixed_point_over_Q=%.17g expected=0 atol=%.1g derivation=stored_input_oracle\n",
                    label, name, err, fp, tol)
        end
        try
            result = equilibrium_moments(m)
            err = norm(result-ref)/norm(ref)
            fp = norm(result-M*result*transpose(M)-Q)/norm(Q)
            @printf("NATIVE %s relative_covariance_error=%.17g fixed_point_over_Q=%.17g expected=0 atol=%.1g derivation=stored_input_oracle\n",
                    label, err, fp, tol)
            failed = !isfinite(err) || !isfinite(fp) || err > tol || fp > tol
            control && failed && error("covariance control failed: $label")
            BAD[] |= failed
        catch ex
            if !control && ex isa ArgumentError && occursin("matrix contains Infs or NaNs", sprint(showerror,ex))
                println("NATIVE ",label," observed=",sprint(showerror,ex),
                        " expected=finite_positive_covariance derivation=stored_input_oracle")
                BAD[] = true
            else
                rethrow()
            end
        end
    end

    function main()
        BLAS.set_num_threads(1)
        @assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
        println("ENV Julia=",VERSION," NonlinearNormalForm=",pkgversion(NNF),
                " SciBmad=",pkgversion(SciBmad)," numerical_threads=1")
        betas, ws = (2.,.5,1.5), (.2,.2,.22)
        M = blocks(.8,ws,betas)
        metrics("ANALYTIC",M,analytic_vectors(.8,ws,betas);control=true)
        F = eigen(M)
        NNF.moveback_unstable!(F)
        metrics("AFTER_PAIR_RECOVERY",M,F)
        for (name,fn) in (("allocating",NNF.mat_eigen),("mutating",NNF.mat_eigen!))
            metrics("DEFAULT $name",M,fn(copy(M)))
            finiteM = blocks(.8,ws,(4.,.25,1.5))
            metrics("SORT_FALSE_PHASE_TRUE $name",finiteM,fn(copy(finiteM);sort=false,phase_modes=true))
        end
        metrics("CONTROL preserved_pairs",M,preserved_pairs(M);control=true)
        finiteM = blocks(.8,ws,(4.,.25,1.5))
        metrics("CONTROL finite_preserved_pairs",finiteM,preserved_pairs(finiteM;sort=false);control=true)
        sigma = covariance((1e-7,1e-7,1e-4),betas)
        J = blocks(0.,(1.,1.,1.),(1.,1.,1.))
        Fcov = NNF.mat_eigen(sigma*J)
        metrics("COVARIANCE_SIGMA_J",sigma*J,Fcov)
        # Geometric emittances are positive imag(eigenvalues(Sigma J)). At beta0*gamma0
        # =10, these fixed modes correspond to normalized (1,1,1000) micrometres.
        modes = sort(abs.(imag(Fcov.values)))[1:2:6]
        @assert isapprox(modes,[1e-7,1e-7,1e-4];rtol=2e-15)
        println("EIGENEMITTANCES normalized_um=",1e7*modes,
                " expected=[1,1,1000] rtol=2e-15 derivation=analytic_block_determinants_beta_gamma_10")
        metrics("CONTROL covariance_preserved_pairs",sigma*J,preserved_pairs(sigma*J);control=true)
        covariance_case("weak_repeated_transport",blocks(.99,(.14,.14,.141),betas),sigma,betas)
        distinct = zeros(6,6)
        for (j,mu) in enumerate((.37,.61,1.13))
            b=betas[j]
            distinct[2j-1:2j,2j-1:2j] = .87*[cos(mu) b*sin(mu); -sin(mu)/b cos(mu)]
        end
        covariance_case("equal_emittances_distinct_tunes",distinct,sigma,betas;control=true)
        covariance_case("split_transport",blocks(.8,(.2,.20001,.22),betas),sigma,betas;control=true)
        println(BAD[] ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
    end
    main()
    '''


    if __name__ == "__main__":
        run_julia(_PROGRAM)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | NonlinearNormalForm 0.5.3 (`7e40539857368faf9784736ad29c8f0ae6e70a23`), managed warm-cache packages; SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d; Julia 1.13.2, CPU, Float64, one thread |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | NonlinearNormalForm 0.5.3 (`7e40539857368faf9784736ad29c8f0ae6e70a23`), managed warm-cache packages; SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d; Julia 1.13.2, CPU, Float64, one thread |

