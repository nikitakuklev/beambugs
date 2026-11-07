---
title: "TRK-23: SaganCavity loses small momentum increments through endpoint subtraction"
hide:
  - toc
---

# TRK-23 — SaganCavity loses small momentum increments through endpoint subtraction

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Discussed in | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/246]] |
| Source | [SciBmad docs/src/track.md:202–214](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/track.md#L202-L214)<br>[docs/src/sagancavity-tracking.md:22–32](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/sagancavity-tracking.md#L22-L32)<br>[BeamTracking src/kernel.jl:18–26](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernel.jl#L18-L26)<br>[149–150](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernel.jl#L149-L150)<br>[src/kernels/sagan_cavity.jl:179–211](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L179-L211)<br>[257–262](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L257-L262)<br>[src/kernels/sagan_cavity.jl:332–345](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L332-L345)<br>[350–352](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L350-L352)<br>[src/kernels/sagan_cavity.jl:74–116](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L74-L116)<br>[src/kernels/rfcavity_kick.jl:52–97](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/rfcavity_kick.jl#L52-L97) |

With Float32 particle coordinates, SaganCavity loses accuracy in small relative momentum changes by subtracting two full momenta. For a proton at Lorentz factor γ=2 receiving 1.000 keV, the stored relative momentum offset δ=(p−p₀)/p₀ is 7.089×10⁻⁷ instead of 7.105×10⁻⁷, understating the gain by 2.339 eV (0.2339%). An independent solution of the relativistic energy–momentum relation establishes the expected value; the existing rationalized reference-momentum helper retains Float32 accuracy. The same subtraction affects fringe energy changes. Scalar and SIMD tracking reproduce the error, while the ordinary RF conversion uses a rationalized expression.

Run `python TRK-23.py`.

??? example "Reproducer: TRK-23.py"

    [Download TRK-23.py](TRK-23.py)

    ```python
    """TRK-23: SaganCavity loses small momentum increments through endpoint subtraction

    With Float32 particle coordinates, SaganCavity loses accuracy in small relative momentum changes by subtracting two full momenta. For a proton at Lorentz factor γ=2 receiving 1.000 keV, the stored relative momentum offset δ=(p−p₀)/p₀ is 7.089×10⁻⁷ instead of 7.105×10⁻⁷, understating the gain by 2.339 eV (0.2339%). An independent solution of the relativistic energy–momentum relation establishes the expected value; the existing rationalized reference-momentum helper retains Float32 accuracy. The same subtraction affects fringe energy changes. Scalar and SIMD tracking reproduce the error, while the ordinary RF conversion uses a rationalized expression.

    Setup: `SCIBMAD_JULIA` set to a Julia launcher, an executable that starts Julia in the SciBmad environment; the Python script runs its embedded Julia program through it.

    The script runs native Julia body and fringe kernels and public scalar, SIMD and zero-length KernelAbstractions CPU tracking in both precisions. A 384-bit Newton solution of the incremental mass-shell equation supplies the expected offset from the exact stored inputs, with zero kicks and the reference helper as controls.
    """

    import os
    import subprocess
    from pathlib import Path

    _PROGRAM = r'''
    using SciBmad, LinearAlgebra, Printf
    const BT = SciBmad.BeamTracking
    BLAS.set_num_threads(1)
    setprecision(BigFloat, 384)
    const SP = Species("proton")
    const M = massof(SP)
    const P = sqrt(3.0)*M

    # With P=p₀c, M=mc² and E=sqrt(P²+M²), the mass shell gives
    # h²+2h=(2E*d+d²)/P² for initial δ=0. Solve for the small h directly;
    # neither full endpoint subtraction nor the upstream helper supplies the oracle.
    function expected_delta(pc, mass, de)
        p, m, d = BigFloat.((pc, mass, de))
        energy = sqrt(p*p+m*m)
        rhs = (2energy*d+d*d)/(p*p)
        h = energy*d/(p*p)
        for _ in 1:12
            h -= (h*h+2h-rhs)/(2+2h)
        end
        @assert abs(h*h+2h-rhs) <= big"1e-108"*max(abs(rhs),big"1e-200")
        return h
    end

    function direct(T, de0; pc=T(P))
        mass, de = T.((M, de0))
        bunch = Bunch(zeros(T,1,6); species=SP, p_over_q_ref=P/C_LIGHT)
        BT.sagan_cavity_kick!(1,bunch.coords,nothing,de,T(2pi*1e8),zero(T),zero(T),mass,pc)
        @assert all(==(BT.STATE_ALIVE),bunch.state) && all(isfinite,bunch.v)
        return bunch.v[1,6]
    end

    function report(label,T,pc,mass,de,observed)
        expected = expected_delta(pc,mass,de)
        err = BigFloat(observed)-expected
        relative = iszero(expected) ? abs(err) : abs(err/expected)
        p, m, d, offset = BigFloat.((pc,mass,de,observed))
        energy = sqrt(p*p+m*m)
        # Rationalize E_out-(E_in+d) to resolve the small energy error as well.
        energy_error = (p*p*(2offset+offset*offset)-2energy*d-d*d) /
            (sqrt(p*p*(1+offset)^2+m*m)+energy+d)
        @printf("%s %s dE_eV=%.12g observed_delta=%.17g expected_delta=%.17g relative_error=%.12g energy_error_eV=%.12g oracle=384-bit_incremental_mass_shell\n",
            label,T,de,observed,Float64(expected),Float64(relative),Float64(energy_error))
        return relative
    end

    function main()
        @assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
        @printf("ENV Julia=%s SciBmad=%s BeamTracking=%s Beamlines=%s Julia_threads=%d BLAS_threads=%d\n",
            VERSION,pkgversion(SciBmad),pkgversion(BT),pkgversion(SciBmad.Beamlines),
            Threads.nthreads(),BLAS.get_num_threads())
        body_bad = false
        fringe_bad = false
        # The decision uses realistic Float32 impulses; the Float64 run is a control.
        for T in (Float32,Float64), de0 in (1000.0,-1000.0,1e6)
            pc, mass, de = T.((P,M,de0))
            observed = direct(T,de0)
            relative = report("BODY",T,pc,mass,de,observed)
            body_bad |= T==Float32 && relative > 48eps(T)
            expected = expected_delta(pc,mass,de)
            stable = BT.dpc_given_dE(pc,de,mass)/pc
            stable_relative = abs((BigFloat(stable)-expected)/expected)
            @assert stable_relative < 8eps(T)
            @printf("HELPER_CONTROL %s dE_eV=%.12g observed_delta=%.17g expected_delta=%.17g relative_error=%.12g oracle=384-bit_incremental_mass_shell\n",
                T,de,stable,Float64(expected),Float64(stable_relative))
        end
        for T in (Float32,Float64)
            observed = direct(T,0)
            @assert iszero(observed)
            println("ZERO_CONTROL ",T," observed_delta=",observed," expected_delta=0 oracle=zero_energy_impulse")
            npublic = 0
            # Three length branches, each scalar/SIMD, plus the zero-length KA CPU path.
            for (ka,simd) in ((false,false),(false,true),(true,false)),
                (len,active) in ((0.0,0.0),(0.4,0.0),(0.4,0.4))
                ka && len!=0 && continue
                elem = RFCavity(L=len,voltage=1000.0,rf_frequency=1e8,phi0=0.0,
                    tracking_method=SaganCavity(n_cells=0,L_active=active))
                branch = Branch([Beamline([Marker()];species_ref=SP,pc_ref=P),
                    Beamline([elem];dE_ref=0.0)])
                line = branch.beamlines[2]
                bunch = Bunch(zeros(T,16,6);species=SP,p_over_q_ref=P/C_LIGHT)
                track!(bunch,line;use_KA=ka,use_explicit_SIMD=simd)
                @assert all(==(BT.STATE_ALIVE),bunch.state) && all(isfinite,bunch.v)
                @assert all(==(bunch.v[1,6]),bunch.v[:,6])
                # Mirror the public lowering of its stored reference, not decimal inputs: the
                # stored p/q can return a momentum one ulp from P, so the kernel gets that value.
                pc = T(sqrt(BT.R_to_E(SP,line.p_over_q_ref)^2-M^2))
                @assert abs(pc-T(P)) <= eps(T(P))
                @assert bunch.v[1,6]==direct(T,1000.0;pc=pc)
                npublic += 1
            end
            @printf("PUBLIC %s cases=%d particles_per_case=16 observed_delta=%.17g expected_delta=%.17g all_alive=true direct_agreement=true oracle=384-bit_incremental_mass_shell\n",
                T,npublic,direct(T,1000.0),Float64(expected_delta(T(P),T(M),T(1000))))
            # Condition the oracle on the native signed fringe impulse. This isolates
            # momentum conversion from the guide's separate fringe-equation mismatch.
            pc, mass = T.((P,M))
            gradient, omega = T.((1e7,2pi*1e8))
            tphi = T(pi/2)/omega
            x = T(0.01)
            de = -(gradient*sin(omega*tphi)*omega/BT.c_light(T))*(x*x)/4
            bunch = Bunch(zeros(T,1,6);species=SP,p_over_q_ref=P/C_LIGHT)
            bunch.v[1,1] = x
            BT.sagan_cavity_fringe!(1,bunch.coords,nothing,gradient,omega,tphi,zero(T),mass,pc,1)
            @assert all(==(BT.STATE_ALIVE),bunch.state) && all(isfinite,bunch.v)
            relative = report("FRINGE",T,pc,mass,de,bunch.v[1,6])
            fringe_bad |= T==Float32 && relative > 48eps(T)
        end
        println(body_bad || fringe_bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
    end

    main()
    '''


    def main() -> None:
        launcher = os.environ.get("SCIBMAD_JULIA")
        if not launcher:
            raise RuntimeError("Set SCIBMAD_JULIA to a Julia launcher that loads SciBmad")
        runner = Path(launcher).expanduser().resolve(strict=True)
        env = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "JULIA_NUM_THREADS",
            "NPROC",
            "PJRT_NPROC",
        ):
            env[name] = "1"
        subprocess.run(
            [str(runner), "--startup-file=no", "-e", _PROGRAM],
            env=env,
            check=True,
            timeout=290,
        )


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | BeamTracking main (`d8e45fdc25dbebf4be8195e69be04df13d7383b1`), pinned source (BeamTracking 0.8.7, SciBmad 0.6.0 5f0fb83ac889, Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float32/Float64, scalar/SIMD, one thread |
| Confirmed<br><small>body kick and fringe momentum conversion</small> | `gpt-6.1-sol` | xhigh | 2026-11-06 | BeamTracking main (`d8e45fdc25dbebf4be8195e69be04df13d7383b1`), pinned source (BeamTracking 0.8.7, SciBmad 0.6.0 5f0fb83ac889, Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float32/Float64, scalar/SIMD and zero-length KA CPU, one thread |

