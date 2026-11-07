---
title: "COL-16: IBS momentum impulses omit the speed-dependent longitudinal conversion"
hide:
  - toc
---

# COL-16 — IBS momentum impulses omit the speed-dependent longitudinal conversion

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [`ibs_damping_and_diffusion!`, ibs_kick.jl:81–126](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/ibs_kick.jl#L81-L126)<br>[ibs_kick.jl:17–35](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/ibs_kick.jl#L17-L35)<br>[`ibs_kick`, general_bl.jl:165–167](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/general_bl.jl#L165-L167)<br>[unpack_bl.jl:156–160](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L156-L160)<br>[`drift`, symplectic_bl.jl:67–71](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/symplectic_bl.jl#L67-L71)<br>[unpack_bl.jl:82–90](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/unpack_bl.jl#L82-L90)<br>[tracking guide, track.md:25–29](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/track.md#L25-L29)<br>[`compute_time`, z_to_time.jl:15–29](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/utils/z_to_time.jl#L15-L29)<br>[`bmad_to_mad!` and `mad_to_bmad!`, rfcavity_kick.jl:47–98](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/rfcavity_kick.jl#L47-L98) |

Deterministic intrabeam-scattering (IBS) impulses change particle speed without rescaling the longitudinal coordinate z, shifting the represented arrival time. The coordinate definition z = βc(t_ref − t), where β is speed divided by the speed of light c, t_ref is reference time and t is arrival time, requires z_out = z_in β_out/β_in at a fixed event. For a 1.230 nC electron bunch at reference kinetic energy 60.32 keV, an impulse assigned to a 7.000 cm drift produces a maximum time jump of 1.115 fs instead of zero. An isolated impulse and independent Cartesian flight calculation agree; disabling IBS preserves time to 9.530 × 10⁻²⁶ s.

Run `julia COL-16.jl`.

??? example "Reproducer: COL-16.jl"

    [Download COL-16.jl](COL-16.jl)

    ```julia
    # COL-16: IBS momentum impulses omit the speed-dependent longitudinal conversion
    #
    # Deterministic intrabeam-scattering (IBS) impulses change particle speed without rescaling the longitudinal coordinate z, shifting the represented arrival time. The coordinate definition z = βc(t_ref − t), where β is speed divided by the speed of light c, t_ref is reference time and t is arrival time, requires z_out = z_in β_out/β_in at a fixed event. For a 1.230 nC electron bunch at reference kinetic energy 60.32 keV, an impulse assigned to a 7.000 cm drift produces a maximum time jump of 1.115 fs instead of zero. An isolated impulse and independent Cartesian flight calculation agree; disabling IBS preserves time to 9.530 × 10⁻²⁶ s.
    #
    # A fixed-seed antithetic Gaussian cloud supplies nonsingular IBS statistics. The normal library builder produces an impulse that is executed without a drift or reference-clock advance; a 256-bit oracle preserves event time using the measured outgoing momentum. Public drift tracking is checked against independent Cartesian flight, with IBS-disabled, zero-z, population and relativistic controls.

    using SciBmad
    using BeamTracking
    using AtomicAndPhysicalConstants
    using LinearAlgebra
    using Random
    using Printf
    using Pkg

    BLAS.set_num_threads(1)
    Threads.nthreads() == 1 || error("Run with JULIA_NUM_THREADS=1")
    const BT = BeamTracking
    const EXT = Base.get_extension(BT, :BeamTrackingBeamlinesExt)
    const C = 299792458.0
    const species = Species("electron")
    const N = 256
    const L = 0.07
    const WEIGHT = 3.0e7

    function cloud()
        rng = MersenneTwister(0xc501)
        widths = [0.8e-3, 0.6e-3, 1.3e-3, 0.9e-3, 6e-3, 1.2e-3]
        a = randn(rng, N ÷ 2, 6) .* permutedims(widths)
        a[1,:] = [0.3e-3, 0.4e-3, -0.7e-3, -0.5e-3, 4.5e-3, -0.8e-3]
        return vcat(a, -a)
    end

    # Independent energy-momentum relation, not a library beta/time helper.
    function beta_big(delta, bg)
        u = BigFloat(bg) * (1 + BigFloat(delta))
        return u / sqrt(1 + u*u)
    end
    function times_big(v, bg, tref)
        return [BigFloat(tref) - BigFloat(v[i,5])/(BigFloat(C)*beta_big(v[i,6], bg))
                for i in axes(v,1)]
    end
    function expected_z(vin, vout, bg)
        return [BigFloat(vin[i,5])*beta_big(vout[i,6], bg)/beta_big(vin[i,6], bg)
                for i in axes(vin,1)]
    end
    function newbunch(v, bg, weight)
        # pc is in eV and chargeof in units of e: R [T*m] = pc/(c*chargeof).
        rigidity = bg*massof(species)/(chargeof(species)*C)
        return Bunch(v=copy(v), species=species, p_over_q_ref=rigidity,
                     weight=weight, t_ref=0.0)
    end

    function impulse(v, bg, weight; damping=true, fluctuations=false)
        bunch = newbunch(v, bg, weight)
        tm = Symplectic(order=2, n_steps=1, ibs_damping_on=damping,
                        ibs_fluctuations_on=fluctuations)
        ref = BT.RefState{Float64}(t_enter=0.0, beta_gamma_enter=bg)
        chain = BT.KernelChain(Val(1), ref)
        chain = EXT.ibs_kick(tm, chain, bunch.p_over_q_ref, bunch, nothing, L)
        call = only(chain.chain)
        call.kernel === BT.ibs_damping_and_diffusion! || error("Unexpected kernel")
        Random.seed!(0xc502)
        for i in axes(v,1)
            call.kernel(i, bunch.coords, call.args...)
        end
        @assert bunch.t_ref == 0.0
        @assert bunch.v[:,[1,3]] == v[:,[1,3]]
        @assert all(isfinite, bunch.v)
        @assert all(==(BT.STATE_ALIVE), bunch.state)
        return bunch
    end

    function public_flight(v, bg, weight; enabled=true, simd=false)
        bunch = newbunch(v, bg, weight)
        tm = Symplectic(order=2, n_steps=1, ibs_damping_on=enabled)
        line = Beamline([Drift(L=L, tracking_method=tm)],
                        species_ref=species, pc_ref=bg*massof(species))
        Random.seed!(0xc502)
        track!(bunch, line; use_explicit_SIMD=simd, use_cpu_multithreading=false)
        @assert all(==(BT.STATE_ALIVE), bunch.state)
        return bunch
    end

    function run_case(label, bg, weight)
        v = cloud()
        kicked = impulse(v, bg, weight)
        ze = expected_z(v, kicked.v, bg)
        tin = times_big(v, bg, 0.0)
        tout = times_big(kicked.v, bg, 0.0)
        dt = tout .- tin
        zerr = BigFloat.(kicked.v[:,5]) .- ze
        k = argmax(abs.(dt))
        @printf("%s bg=%.8g kinetic_eV=%.17g charge_C=%.17g max_abs_delta_kick=%.17g\n",
                label, bg, massof(species)*(sqrt(1+bg*bg)-1), N*weight*1.602176634e-19,
                maximum(abs.(kicked.v[:,6] .- v[:,6])))
        @printf("%s probe=1 delta_in=%.17g delta_out=%.17g beta_in=%.17g beta_out=%.17g z_in=%.17g z_observed=%.17g z_expected=%.17g delta_t_s=%.17g expected_delta_t_s=0 (fixed event: z_out=z_in*beta_out/beta_in)\n",
                label, v[1,6], kicked.v[1,6], beta_big(v[1,6],bg),
                beta_big(kicked.v[1,6],bg), v[1,5], kicked.v[1,5], ze[1], dt[1])
        @printf("%s worst_row=%d delta_in=%.17g delta_out=%.17g z_in=%.17g z_expected=%.17g max_abs_time_jump_s=%.17g max_abs_z_error_m=%.17g\n",
                label, k, v[k,6], kicked.v[k,6], v[k,5], ze[k], maximum(abs.(dt)), maximum(abs.(zerr)))

        # The library's actual bmad<->mad coordinate transforms provide a second
        # check: apply the observed outgoing momentum at fixed tau = c(tref-t).
        converted = newbunch(v, bg, weight)
        beta0 = bg/sqrt(1+bg*bg)
        for i in axes(v,1)
            BT.bmad_to_mad!(i, converted.coords, beta0, 1/bg, 0)
            p = kicked.v[i,6]
            y = beta0*(2*p+p*p)
            converted.v[i,6] = y/(1+sqrt(1+beta0*y))
            BT.mad_to_bmad!(i, converted.coords, beta0, 1/bg, 0)
        end
        conversion_error = maximum(abs.(BigFloat.(converted.v[:,5]) .- ze))
        @assert conversion_error < 3e-17
        @printf("%s library_fixed_tau_conversion_max_z_error_m=%.17g\n", label, conversion_error)

        # Do not infer the impulse by subtracting a drift. Instead propagate the
        # measured outgoing physical momenta along straight Cartesian worldlines.
        # ps is momentum along the axis / p_ref; speed_s = beta*c*ps/(1+delta).
        public = public_flight(v, bg, weight)
        @assert maximum(abs.(public.v[:,[2,4,6]] .- kicked.v[:,[2,4,6]])) < 2e-17
        expected_time = [tin[i] + BigFloat(L)*(1+BigFloat(kicked.v[i,6])) /
                         (BigFloat(C)*beta_big(kicked.v[i,6],bg)*
                          sqrt((1+BigFloat(kicked.v[i,6]))^2 -
                               BigFloat(kicked.v[i,2])^2 - BigFloat(kicked.v[i,4])^2))
                         for i in axes(v,1)]
        observed_time = times_big(public.v, bg, public.t_ref)
        flight_error = observed_time .- expected_time
        # The full public result must contain exactly the separately measured impulse error.
        # This identity also holds when a corrected impulse preserves time.
        closure = maximum(abs.(flight_error .- dt))
        @assert closure < 2e-24
        @printf("%s public_max_time_error_s=%.17g impulse_flight_closure_s=%.17g public_probe_time_observed_s=%.17g public_probe_time_expected_s=%.17g\n",
                label, maximum(abs.(flight_error)), closure, observed_time[1], expected_time[1])
        if label == "main"
            simd = public_flight(v, bg, weight; simd=true)
            @assert maximum(abs.(simd.v .- public.v)) < 3e-17
            @printf("simd_max_coordinate_difference=%.17g\n", maximum(abs.(simd.v .- public.v)))
            # Controls have no kick and z=0 at the impulse respectively.
            no_kick = impulse(v, bg, weight; damping=false)
            @assert no_kick.v == v
            no_public = public_flight(v, bg, weight; enabled=false)
            no_t = times_big(no_public.v, bg, no_public.t_ref)
            no_expected = [tin[i] + BigFloat(L)*(1+BigFloat(v[i,6])) /
                           (BigFloat(C)*beta_big(v[i,6],bg)*
                            sqrt((1+BigFloat(v[i,6]))^2 - BigFloat(v[i,2])^2 - BigFloat(v[i,4])^2))
                           for i in axes(v,1)]
            control_error = maximum(abs.(no_t .- no_expected))
            @assert control_error < 2e-24
            vzero = copy(v)
            vzero[1,5] = 0.0
            zero_kick = impulse(vzero, bg, weight)
            @assert times_big(zero_kick.v,bg,0.0)[1] == times_big(vzero,bg,0.0)[1] == 0
            @assert zero_kick.v[1,6] != vzero[1,6]
            @printf("controls no_kick_impulse_time_jump_s=0 no_IBS_public_max_time_error_s=%.17g z_zero_time_jump_s=0 z_zero_delta_kick=%.17g\n",
                    control_error, zero_kick.v[1,6]-vzero[1,6])
        end
        return maximum(abs.(dt))
    end

    println("Julia=", VERSION, " threads=", Threads.nthreads(), " BLAS_threads=", BLAS.get_num_threads())
    for (uuid, info) in sort(collect(Pkg.dependencies()); by=x->x[2].name)
        if info.name in ("SciBmad", "BeamTracking", "Beamlines", "AtomicAndPhysicalConstants")
            println("PACKAGE name=",info.name," version=",info.version," tree=",info.tree_hash," source=",info.source)
        end
    end
    setprecision(BigFloat,256) do
        main_error = run_case("main", 0.5, WEIGHT)
        run_case("population_div100", 0.5, WEIGHT/100)
        run_case("relativistic", 20.0, WEIGHT)
        println(main_error > 1e-17 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
    end

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

