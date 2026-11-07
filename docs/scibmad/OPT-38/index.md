---
title: "OPT-38: Equal-tune periodic Twiss aborts on a higher-order zero resonance divisor"
hide:
  - toc
---

# OPT-38 — Equal-tune periodic Twiss aborts on a higher-order zero resonance divisor

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on NonlinearNormalForm commit [`dfc5c26`](https://github.com/bmad-sim/NonlinearNormalForm.jl/commit/dfc5c2697e53), committed 2026-11-06 |
| Discussed in | [[gh:https://github.com/bmad-sim/NonlinearNormalForm.jl/issues/65]] |
| Source | [`normal`, `src/normal.jl:165–179`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L165-L179)<br>[`res=nothing`, line 8](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L8)<br>[`is_orbital_resonance`, lines 463–488](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L463-L488)<br>[`src/normal.jl:100–108`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L100-L108)<br>[`src/matrix.jl:124–150`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/matrix.jl#L124-L150)<br>[`src/twiss/twiss.jl:58–108`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L58-L108)<br>[`twiss` signature, lines 198–239](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L204-L245)<br>[periodic descriptor paths, lines 319–325](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L325-L331)<br>[`_a_r_tunes`, lines 512–519](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L518-L525)<br>[`ext/BeamTrackingBeamlinesExt/symplectic_bl.jl:265–274`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L265-L274)<br>[body rotations, `src/kernels/quadrupole_kick.jl:44–60`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/quadrupole_kick.jl#L44-L60) |

For stable, uncoupled FODO rings with equal transverse tunes, periodic twiss at second Taylor order can terminate the Julia process with GTPSA invsqrt(NaN). Roundoff-sized cross-plane terms are divided by a zero resonance denominator during nonlinear normalization, although the linear eigenvectors remain finite and correctly paired. A 1 GeV/c proton cell with quadrupole strengths ±1.600 m⁻², 0.3000 m magnets and 1.700 m drifts has tune 0.1505 in both planes and finite beta functions 6.848 m and 2.825 m. Independent analytic normalization establishes these values; retaining the resonance in the underlying normal-form calculation prevents the abort.

Run `python OPT-38.py` with [_scibmad.py](../helpers/_scibmad.py) in the same directory.

??? example "Reproducer: OPT-38.py"

    [Download OPT-38.py](OPT-38.py)

    ```python
    """OPT-38: Equal-tune periodic Twiss aborts on a higher-order zero resonance divisor

    For stable, uncoupled FODO rings with equal transverse tunes, periodic twiss at second Taylor order can terminate the Julia process with GTPSA invsqrt(NaN). Roundoff-sized cross-plane terms are divided by a zero resonance denominator during nonlinear normalization, although the linear eigenvectors remain finite and correctly paired. A 1 GeV/c proton cell with quadrupole strengths ±1.600 m⁻², 0.3000 m magnets and 1.700 m drifts has tune 0.1505 in both planes and finite beta functions 6.848 m and 2.825 m. Independent analytic normalization establishes these values; retaining the resonance in the underlying normal-form calculation prevents the abort.

    Setup: `SCIBMAD_JULIA` set to a Julia launcher, an executable that starts Julia in the SciBmad environment, and `taskset` on `PATH` with CPU 0 among the allowed CPUs; the script runs each case as a child process pinned to CPU 0, which runs its embedded Julia program through the launcher.

    The script launches one managed Julia process per case and compares periodic optics with independent 256-bit hard-edge matrices. It checks finite analytic momentum-dependent normalizers, the tracked map and its linear eigenvectors, and resonance retention in the underlying normal-form calculation. First-order, split-tune, weaker equal-tune and open-propagation cases provide controls; preserving eigenvector pairs alone still aborts. All cases use fixed inputs and one numerical thread.
    """

    import os
    import subprocess
    import sys
    from pathlib import Path

    from _scibmad import run_julia

    _PROGRAM = r'''
    using SciBmad, LinearAlgebra, Printf
    using SciBmad.GTPSA
    const NNF = SciBmad.NNF
    const TI = SciBmad.TI
    BLAS.set_num_threads(1)
    @assert Threads.nthreads() == BLAS.get_num_threads() == 1
    mode, order_text, chrom_text, variant = split(ENV["SCIBMAD_REPRO_CASE"], ",")
    order, chrom = parse(Int, order_text), parse(Int, chrom_text)
    k, L, dl = variant == "weak" ? (0.37, 0.5, 1.7) : (1.6, 0.3, 1.7)
    split_strength = variant == "split" ? 1e-5 : 0.0

    # Canonical momenta are divided by reference momentum, r=1+delta.
    # Exact uncoupled hard-edge blocks give M=D*Qd*D*Qf in traversal order.
    function solution(sgn, delta)
        r = 1 + delta
        kb, lb, db = BigFloat(k), BigFloat(L), BigFloat(dl)
        function q(kk)
            omega = sqrt(abs(kk)/r)
            c = kk > 0 ? cos(omega*lb) : cosh(omega*lb)
            s = kk > 0 ? sin(omega*lb) : sinh(omega*lb)
            [c s/(r*omega); -sign(kk)*r*omega*s c]
        end
        Qf, Qd = q(sgn*kb), q(-sgn*kb*BigFloat(1+split_strength))
        D = [one(r) db/r; zero(r) one(r)]
        M = D*Qd*D*Qf
        mu = acos(tr(M)/2)
        beta = M[1,2]/sin(mu)
        alpha = (M[1,1]-M[2,2])/(2sin(mu))
        A = [sqrt(beta) zero(r); -alpha/sqrt(beta) 1/sqrt(beta)]
        R = [cos(mu) sin(mu); -sin(mu) cos(mu)]
        exit_map = Qf*M*inv(Qf)
        (; M, mu, beta, alpha, A, R, beta_exit=exit_map[1,2]/sin(mu))
    end
    setprecision(BigFloat, 256)
    solutions = [solution(sgn, big"0") for sgn in (1,-1)]
    beta_refs = [Float64(s.beta_exit) for s in solutions]
    Mref = zeros(4,4)
    for plane in 1:2
        idx = 2plane-1:2plane
        Mref[idx,idx] = Float64.(solutions[plane].M)
    end

    function nonfinite(v)
        mono = zeros(UInt8, 6)
        value = Ref{TI.numtype(eltype(v))}()
        count_bad = 0
        for t in v
            idx = GTPSA.cycle!(t, -1, 6, mono, value)
            while idx >= 0
                count_bad += !isfinite(value[])
                idx = GTPSA.cycle!(t, idx, 6, mono, value)
            end
        end
        count_bad
    end

    if mode == "pairs"
        # Preserve LAPACK conjugate columns while retaining native normalization.
        # This isolates the earlier eigenvector-pairing defect from this divisor.
        @eval NNF function mat_eigen(mat; sort=true, phase_modes=true, error_unstable=true)
            F = make_mutable(eigen(mat; sortby=nothing))
            low_mat_eigen!(F, sort, phase_modes, error_unstable)
            F
        end
    end
    bl = Beamline([
        Quadrupole(L=L, Kn1=k), Drift(L=dl),
        Quadrupole(L=L, Kn1=-k*(1+split_strength)), Drift(L=dl)
    ], species_ref=Species("proton"), pc_ref=1e9)
    # The parent accepts a native domain exit only after this marker, so import,
    # construction and independent-oracle failures cannot count as reproduction.
    println("READY mode=",mode," order=",order," chrom=",chrom," variant=",variant)
    flush(stdout)
    if mode == "normal"
        println("ENV Julia=",VERSION," SciBmad=",pkgversion(SciBmad),
                " BeamTracking=",pkgversion(SciBmad.BeamTracking),
                " Beamlines=",pkgversion(SciBmad.Beamlines),
                " NonlinearNormalForm=",pkgversion(NNF)," GTPSA=",pkgversion(GTPSA),
                " numerical_threads=1")
        for (sgn, label) in ((1,"x"),(-1,"y"))
            s = solution(sgn, big"0")
            h = big"1e-20"
            Ap = (solution(sgn,h).A-solution(sgn,-h).A)/(2h)
            # A(delta) has a canonical coasting extension with quadratic z term
            # -u^T A^T J (dA/ddelta) u / 2; its coefficient matrix is symmetric.
            J = BigFloat[0 1; -1 0]
            Z = -transpose(s.A)*J*Ap
            residual = maximum(abs, inv(s.A)*s.M*s.A-s.R)
            symmetry_error = maximum(abs, Z-transpose(Z))
            @assert residual < big"1e-74" && abs(det(s.A)-1) < big"1e-74"
            @assert all(isfinite,Ap) && symmetry_error < big"1e-38"
            @assert abs(s.mu-solution(-sgn,big"0").mu) < big"1e-74"
            @printf("ANALYTIC %s tune=%.16g beta_m=%.16g normalizer_residual=%.4g expected=0 atol=1e-74 A_delta_max=%.4g z_symmetry_error=%.4g expected=0 atol=1e-38 derivation=256_bit_hard_edge_Courant_Snyder\n",
                    label, Float64(s.mu/(2big(pi))), Float64(s.beta),
                    Float64(residual), Float64(maximum(abs,Ap)), Float64(symmetry_error))
        end
        X = vars(Descriptor([2,2,2,2,2,2],2))
        b = Bunch(v=reshape(copy.(X),1,6), species=Species("proton"),
                  p_over_q_ref=1e9/C_LIGHT)
        track!(b,bl; rf_on=false,use_explicit_SIMD=false)
        m = DAMap(v=vec(copy.(b.v)),v0=zeros(5),nv=5,np=1)
        M = Matrix(NNF.jacobian(m,NNF.HVARS))
        map_error = maximum(abs,M-Mref)
        @assert nonfinite(vec(b.v)) == 0 && map_error < 2e-14
        @printf("MAP nonfinite=0 max_matrix_error=%.4g expected=0 atol=2e-14 cross_plane_max=%.4g expected=0 derivation=analytic_separable_hard_edge_blocks\n",
                map_error, maximum(abs,[M[1:2,3:4];M[3:4,1:2]]))
        Mt = Matrix(transpose(M))
        F = NNF.mat_eigen(Mt; phase_modes=false)
        pair_error = maximum(norm(F.vectors[:,2j-1]-conj(F.vectors[:,2j])) for j in 1:2)
        eigen_error = norm(Mt*F.vectors-F.vectors*Diagonal(F.values))/norm(Mt)
        @assert all(isfinite,F.vectors) && rank(F.vectors)==4
        @assert pair_error < 2e-14 && eigen_error < 2e-14
        @printf("LINEAR_NORMALIZATION rank=4 expected=4 conjugacy_error=%.4g eigenvector_residual=%.4g expected=0 atol=2e-14 derivation=Mt_V_equals_V_Lambda\n",pair_error,eigen_error)
        native_bad = nonfinite(normal(m).v)
        println("NATIVE_NORMAL nonfinite=",native_bad," expected=0 derivation=finite_analytic_normalizer")
        a = normal(m; res=reshape([1,-1],2,1))
        retained_bad = nonfinite(a.v)
        symp_error = norm(NNF.checksymp(NNF.jacobian(a,NNF.HVARS)))
        fa = factorise(a;canonise=1)
        @assert retained_bad == 0 && nonfinite(fa.a.v) == 0 && symp_error < 2e-14
        @printf("RETAIN_RESONANCE nonfinite=%d factorization_nonfinite=0 linear_symplectic_residual=%.4g expected=0 atol=2e-14 derivation=retain_1_minus1_resonance\n",retained_bad,symp_error)
    else
        kwargs = (rf_on=false,order=order,chrom=chrom,at=[1],cols=["beta1","beta2"])
        if mode == "open"
            X = vars(Descriptor([order,order,order,order,order,chrom],max(order,chrom)))
            ray = copy.(X)
            for plane in 1:2
                idx = 2plane-1:2plane
                ray[idx] = Float64.(solutions[plane].A)*X[idx]
            end
            a = DAMap(v=ray,v0=zeros(5),nv=5,np=1)
            kwargs = merge(kwargs,(a_initial=a,))
        end
        tw = twiss(bl;kwargs...)
        observed = [only(tw.beta1),only(tw.beta2)]
        relative_error = maximum(abs,(observed-beta_refs)./beta_refs)
        @assert all(isfinite,observed) && relative_error < 5e-14
        @printf("TWISS beta_m=(%.16g,%.16g) expected=(%.16g,%.16g) relative_error=%.4g rtol=5e-14 derivation=256_bit_hard_edge_matrices\n",
                observed...,beta_refs...,relative_error)
    end
    println("CASE COMPLETED")
    '''

    _CASES = (
        ("first_order", "twiss,1,1,base", False),
        ("order2", "twiss,2,2,base", True),
        ("order3", "twiss,3,3,base", True),
        ("order2_chrom1", "twiss,2,1,base", True),
        ("order1_chrom2", "twiss,1,2,base", True),
        ("preserved_pairs", "pairs,2,2,base", True),
        ("split_tunes", "twiss,2,2,split", False),
        ("weak_equal_tunes", "twiss,2,2,weak", False),
        ("open_order2", "open,2,2,base", False),
        ("normalizer", "normal,2,2,base", False),
    )


    def main() -> None:
        """Recognize the native abort and propagate unrelated execution failures."""
        defect = False
        for label, case, can_abort in _CASES:
            env = os.environ.copy()
            env["SCIBMAD_REPRO_CASE"] = case
            result = subprocess.run(
                ["taskset", "-c", "0", sys.executable, str(Path(__file__).resolve()), "--case"],
                env=env, capture_output=True, text=True, check=False, timeout=290,
            )
            output = result.stdout + result.stderr
            domain_abort = (
                result.returncode == 1
                and "READY " in result.stdout
                and "invalid domain invsqrt(" in output
                and "NAN" in output.upper()
            )
            if can_abort and domain_abort:
                defect = True
                print(f"{label} exit=1 observed=GTPSA_invsqrt(NaN)_process_termination "
                      "expected=finite_beta_m_(6.848107428164884,2.824809954876377) "
                      "derivation=analytic_separable_hard_edge_normalization", flush=True)
            else:
                if result.returncode != 0 or "CASE COMPLETED" not in result.stdout:
                    raise RuntimeError(f"{label}: execution failed ({result.returncode})\n{output}")
                for line in result.stdout.splitlines():
                    if line.startswith(("READY ", "CASE COMPLETED")):
                        continue
                    print(f"{label} {line}", flush=True)
                    if line.startswith("NATIVE_NORMAL nonfinite="):
                        defect |= line.split()[1] != "nonfinite=0"
                if result.stderr:
                    print(result.stderr, file=sys.stderr, end="", flush=True)
        print("DEFECT REPRODUCED" if defect else "NOT REPRODUCED", flush=True)


    if __name__ == "__main__":
        if sys.argv[1:] == ["--case"]:
            try:
                run_julia(_PROGRAM)
            except subprocess.CalledProcessError as exc:
                sys.exit(exc.returncode)
        elif sys.argv[1:]:
            raise ValueError("Run without arguments")
        else:
            main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-06 | NonlinearNormalForm 0.5.3 (`7e40539857368faf9784736ad29c8f0ae6e70a23`), managed warm-cache packages; SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d, GTPSA a5790a7deaa6; Julia 1.13.2, CPU, Float64, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | NonlinearNormalForm 0.5.3 (`7e40539857368faf9784736ad29c8f0ae6e70a23`), managed warm-cache packages; SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d, GTPSA a5790a7deaa6; Julia 1.13.2, CPU, Float64, one thread |

