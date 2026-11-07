---
title: "TRK-24: Long absolute clocks erase particle time offsets before parameter evaluation"
hide:
  - toc
---

# TRK-24 — Long absolute clocks erase particle time offsets before parameter evaluation

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`SciBmad docs/src/timedependent.md:25–44`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/timedependent.md#L25-L44)<br>[`src/utils/z_to_time.jl:15–29`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/utils/z_to_time.jl#L15-L29)<br>[`src/kernel.jl:239–246`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernel.jl#L239-L246)<br>[`src/kernel.jl:217–222`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernel.jl#L217-L222)<br>[`ext/BeamTrackingBeamlinesExt/sagan_cavity_bl.jl:15–19`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/sagan_cavity_bl.jl#L15-L19)<br>[`src/kernels/sagan_cavity.jl:179–211`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L179-L211) |

Fast time-dependent element parameters lose particle arrival-time resolution when evaluated against a long absolute reference clock. The conversion forms one floating-point absolute time before calling the user's function, so subtracting a clock origin inside that function cannot recover the offset. With a 50-hour clock, a zero-length RF kick whose voltage is modulated at 400.8 MHz gains 0.9997 MeV instead of 0.9871 MeV for a 10.00 ps early arrival, a 1.275% error. An analytic thin-kicker calculation independently confirms the timing mechanism. Constant-voltage relative RF remains accurate, and short clocks or slow parameter variation have much smaller errors.

Run `python TRK-24.py` with [_scibmad.py](../helpers/_scibmad.py) in the same directory.

??? example "Reproducer: TRK-24.py"

    [Download TRK-24.py](TRK-24.py)

    ```python
    """TRK-24: Long absolute clocks erase particle time offsets before parameter evaluation

    Fast time-dependent element parameters lose particle arrival-time resolution when evaluated against a long absolute reference clock. The conversion forms one floating-point absolute time before calling the user's function, so subtracting a clock origin inside that function cannot recover the offset. With a 50-hour clock, a zero-length RF kick whose voltage is modulated at 400.8 MHz gains 0.9997 MeV instead of 0.9871 MeV for a 10.00 ps early arrival, a 1.275% error. An analytic thin-kicker calculation independently confirms the timing mechanism. Constant-voltage relative RF remains accurate, and short clocks or slow parameter variation have much smaller errors.

    Setup: `SCIBMAD_JULIA` set to a Julia launcher, an executable that starts Julia in the SciBmad environment; the Python script runs its embedded Julia program through it.

    The script tracks fixed proton bunches through zero-length RF cavities and an independent thin horizontal kicker. A 256-bit thin-impulse oracle retains the particle offset, and a second oracle predicts the result after one absolute-time rounding; short clocks, slow modulation and constant voltage provide controls. Scalar and SIMD cases use one numerical thread.
    """

    from _scibmad import run_julia

    _PROGRAM = r'''
    using SciBmad, LinearAlgebra, Printf, SpaceCharge
    const BT=SciBmad.BeamTracking
    const BAD=Ref(false)
    BLAS.set_num_threads(1); SpaceCharge.FFTW.set_num_threads(1)
    setprecision(BigFloat,256)
    const sp=Species("proton")
    const mass=massof(sp)
    const pc=sqrt(3.0)*mass
    const velocity=(pc/mass)/sqrt(1+(pc/mass)^2)*C_LIGHT
    @assert Threads.nthreads()==BLAS.get_num_threads()==SpaceCharge.FFTW.get_num_threads()==1
    println("ENV Julia=",VERSION," SciBmad=",pkgversion(SciBmad)," BeamTracking=",pkgversion(BT)," Beamlines=",pkgversion(SciBmad.Beamlines)," numerical_threads=1")
    function track_rf(A,fc,fm,tau;simd=false,dynamic=true)
        z=velocity*tau
        omega_c=2pi*fc; omega_m=2pi*fm
        V=dynamic ? 1e6*(1+0.5sin(omega_m*(Time()-A))) : 1e6
        rf=RFCavity(L=0.,voltage=V,rf_frequency=fc,phi0=0.,zero_phase=PhaseRef.Accelerating,tracking_method=SaganCavity(n_cells=0,L_active=0.))
        branch=Branch([Beamline([Marker()];species_ref=sp,pc_ref=pc),Beamline([rf];dE_ref=0.)])
        b=Bunch(repeat([0. 0. 0. 0. z 0.],16,1);species=sp,p_over_q_ref=pc/C_LIGHT,t_ref=A)
        track!(b,branch.beamlines[2];use_explicit_SIMD=simd)
        @assert all(==(BT.STATE_ALIVE),b.state)
        p=BigFloat(pc); m=BigFloat(mass); E0=sqrt(p^2+m^2)
        # The momentum is the measured output; solve the mass shell at high precision.
        d=BigFloat(b.v[1,6]); actual=sqrt(p^2*(1+d)^2+m^2)-E0
        dt=-BigFloat(z)/(p/E0*BigFloat(C_LIGHT))
        voltage=dynamic ? BigFloat(1e6)*(1+BigFloat(0.5)*sin(BigFloat(omega_m)*dt)) : BigFloat(1e6)
        expected=voltage*cos(BigFloat(omega_c)*dt)
        delta_expected=sqrt((E0+expected)^2-m^2)/p-1
        # A separate check rounds the absolute arrival time once, before the function.
        rounded=Float64(BigFloat(A)-BigFloat(z)/BigFloat(velocity))
        sampled=BT.compute_time(z,0.,A,pc/mass)
        dts=BigFloat(rounded)-BigFloat(A)
        predicted_voltage=dynamic ? BigFloat(1e6)*(1+BigFloat(0.5)*sin(BigFloat(omega_m)*dts)) : BigFloat(1e6)
        predicted=predicted_voltage*cos(BigFloat(omega_c)*dt)
        # Compare with both oracles so an implementation preserving the offset
        # can clear the defect while an unrelated tracking failure stops the run.
        @assert min(abs(actual-predicted),abs(actual-expected))<2e-6 (A,fc,fm,tau,actual,predicted)
        if A >= 43200 && dynamic && fm == 400.8e6
            BAD[] |= abs(actual-expected)>2e-6
        elseif !dynamic || A <= 1e-5
            @assert abs(actual-expected)<2e-6
        else
            # The waveform slope bounds the effect of half an absolute-time ulp.
            rounding_budget=BigFloat(1e6)*BigFloat(0.5)*abs(BigFloat(omega_m))*BigFloat(eps(A))/2
            @assert abs(actual-expected)<rounding_budget+big"2e-6"
        end
        @printf("RF A=%.17g fc=%.17g fm=%.17g tau=%.17g SIMD=%s dynamic=%s ulp_s=%.17g sampled_dt=%.17g true_dt=%.17g observed_eV=%.17g expected_eV=%.17g error_eV=%.17g rel_error=%.17g rounded_prediction_error_eV=%.3g delta_error=%.3g derivation=256_bit_thin_RF_impulse_and_mass_shell\n",A,fc,fm,tau,simd,dynamic,eps(A),Float64(BigFloat(sampled)-BigFloat(A)),Float64(dt),Float64(actual),Float64(expected),Float64(actual-expected),Float64((actual-expected)/expected),Float64(actual-predicted),Float64(d-delta_expected))
    end
    function kicker(A,f,tau;simd=false)
        z=velocity*tau
        omega=2pi*f
        ele=HKicker(L=0.,Kn0L=1e-4*(1+0.5sin(omega*(Time()-A))))
        bl=Beamline([ele];species_ref=sp,pc_ref=pc)
        b=Bunch(repeat([0. 0. 0. 0. z 0.],16,1);species=sp,p_over_q_ref=pc/C_LIGHT,t_ref=A)
        track!(b,bl;use_explicit_SIMD=simd)
        dt=-BigFloat(z)/(BigFloat(pc)/sqrt(BigFloat(pc)^2+BigFloat(mass)^2)*BigFloat(C_LIGHT))
        expected=-BigFloat(1e-4)*(1+BigFloat(0.5)*sin(BigFloat(omega)*dt))
        observed=b.v[1,2]
        sampled=BT.compute_time(z,0.,A,pc/mass)
        rounded_expected=-1e-4*(1+0.5sin(omega*(sampled-A)))
        @assert min(abs(observed-rounded_expected),abs(observed-expected))<2e-19
        A >= 43200 && (BAD[] |= abs(observed-expected)>2e-19)
        @printf("KICKER A=%.17g f=%.17g tau=%.17g SIMD=%s observed_px=%.17g expected_px=%.17g rel_error=%.17g derivation=analytic_thin_kicker_Dpx_minus_Kn0L\n",A,f,tau,simd,observed,Float64(expected),Float64((observed-expected)/expected))
    end

    function main()
        # Assign the absolute clock directly: this tests sampling, not accumulated drift.
        f = 400.8e6
        for A in (0.,1e-5,1e6/11245.,43200.,180000.)
            track_rf(A,f,f,1/(16f))
        end
        for simd in (false,true), (A,tau) in ((0.,10e-12),(43200.,3e-12),(180000.,10e-12))
            track_rf(A,f,f,tau;simd)
            kicker(A,f,tau;simd)
        end
        for simd in (false,true)
            track_rf(180000.,f,f,10e-12;simd,dynamic=false)
        end
        # For a slow waveform, the voltage slope times half the absolute-time ulp
        # bounds the loss. The same clock and particle offset therefore remain useful.
        track_rf(180000.,f,1e3,10e-12)
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
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | BeamTracking main (`d8e45fdc25dbebf4be8195e69be04df13d7383b1`), source (BeamTracking 0.8.7, SciBmad 0.6.0 5f0fb83ac889, Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float64, scalar/SIMD, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | BeamTracking main (`d8e45fdc25dbebf4be8195e69be04df13d7383b1`), source (BeamTracking 0.8.7, SciBmad 0.6.0 5f0fb83ac889, Beamlines 0.11.0 35fe0162c72d), Julia 1.13.2, CPU, Float64, scalar/SIMD, one thread |

