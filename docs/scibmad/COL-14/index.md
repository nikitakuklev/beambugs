---
title: "COL-14: Sagan RF spin impulses omit the superimposed solenoid vector potential"
hide:
  - toc
---

# COL-14 — Sagan RF spin impulses omit the superimposed solenoid vector potential

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on BeamTracking commit [`0bf3585`](https://github.com/bmad-sim/BeamTracking.jl/commit/0bf3585fdc9c), committed 2026-10-28 |
| Source | [`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/sagan_cavity_bl.jl:38-50`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/sagan_cavity_bl.jl#L38-L50)<br>[`src/kernels/sagan_cavity.jl:35-57`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L35-L57)<br>[`sagan_cavity.jl:179-216`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L179-L216)<br>[`sagan_cavity.jl:196-198,216`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/sagan_cavity.jl#L196-L216)<br>[`src/kernels/spin.jl:79-80`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/kernels/spin.jl#L79-L80)<br>[`SciBmad.jl/docs/src/sagancavity-physics.md:9,32-35`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/sagancavity-physics.md#L9-L35)<br>[`sagancavity-physics.md:69`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/sagancavity-physics.md#L69)<br>[`166-167`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/sagancavity-physics.md#L166-L167)<br>[`sagancavity-physics.md:104-107`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/sagancavity-physics.md#L104-L107) |

A `SaganCavity` with a superimposed solenoid field and zero active RF length (`L_active=0`) uses canonical transverse momenta directly in its RF spin kick instead of first subtracting the solenoid vector potential. By the Thomas–BMT equation, a longitudinal electric field gives no spin rotation for longitudinal mechanical motion, and independent high-precision solenoid transport preserves an initially longitudinal spin. For off-axis protons initially moving and polarized longitudinally, with P₀c (reference momentum times light speed) twice the rest energy, a 0.4300 m element of normalized solenoid strength 1.400 m⁻¹ and RF energy gain 1.877 keV produces a maximum transverse spin of 2.727 × 10⁻⁷ instead of zero. Zero-solenoid and pure-solenoid controls pass.

Run `julia COL-14.jl`.

??? example "Reproducer: COL-14.jl"

    [Download COL-14.jl](COL-14.jl)

    ```julia
    # COL-14: Sagan RF spin impulses omit the superimposed solenoid vector potential
    #
    # A `SaganCavity` with a superimposed solenoid field and zero active RF length (`L_active=0`) uses
    # canonical transverse momenta directly in its RF spin kick instead of first subtracting the
    # solenoid vector potential. By the Thomas–BMT equation, a longitudinal electric field gives no spin
    # rotation for longitudinal mechanical motion, and independent high-precision solenoid transport
    # preserves an initially longitudinal spin. For off-axis protons initially moving and polarized
    # longitudinally, with P₀c (reference momentum times light speed) twice the rest energy, a 0.4300 m
    # element of normalized solenoid strength 1.400 m⁻¹ and RF energy gain 1.877 keV produces a maximum
    # transverse spin of 2.727 × 10⁻⁷ instead of zero. Zero-solenoid and pure-solenoid controls pass.
    #
    # The script tracks 16 off-axis particles whose canonical momenta equal the vector potential, so
    # that they move longitudinally, through Sagan RF cavities with a superimposed solenoid, sweeping
    # species, energy, solenoid strength, RF gain, length and the scalar and SIMD paths, and compares
    # the spin with a 256-bit oracle of the pure solenoid rotation; zero-field and zero-gain cases are
    # controls. A second test translates moving particles in a way that preserves mechanical momentum,
    # which must not change their spin. Local copies of the RF impulse show that halving its spin
    # strength alone leaves the error, while supplying the vector potential removes it.
    using SciBmad, BeamTracking, LinearAlgebra, Printf
    BLAS.set_num_threads(1)

    # These local impulse controls retain the pinned orbital arithmetic. The
    # first changes both spin lengths to one half; the second also supplies A.
    # They operate only in this module and never modify the loaded package.
    module ControlKicks
    using BeamTracking: Coords, STATE_ALIVE, STATE_LOST_PZ, PZI, ZI, XI, YI,
        c_light, rotate_spin_field!, to_energy_coords!, to_momentum_coords!, vifelse
    function half_kick!(i, coords::Coords, 
                                a, q_voltage, rf_omega, t_phi0, t_ref, mass, P0c)
      v = coords.v
      alive = (coords.state[i] == STATE_ALIVE)

      pz = v[i,PZI]
      Pc = (1 + pz) * P0c
      m_over_pc = mass / ((1 + v[i,PZI]) * P0c)
      beta = 1 / sqrt(1 + m_over_pc * m_over_pc)
      t = t_phi0 + t_ref - v[i,ZI] / (beta * c_light(eltype(coords.v)))
      dE = q_voltage * cos(rf_omega * t)

      # 

      if !isnothing(coords.q)
        e_field = (0, 0, dE * c_light(eltype(coords.v)) / P0c)
        b_field = (0, 0, 0)
        a_potential = 0
        g_bend = 0
        rotate_spin_field!(i, coords, a, g_bend, mass/P0c, a_potential, a_potential, e_field, b_field, 0.5)
      end

      # Energy kick

      to_energy_coords!(i, coords, beta)
      rad = dE*dE + 2*sqrt(Pc*Pc + mass^2) * dE + Pc*Pc
      coords.state[i] = vifelse(rad < 0, STATE_LOST_PZ, coords.state[i])
      alive = (coords.state[i] == STATE_ALIVE)
      sqrt_rad = vifelse(alive, sqrt(abs(rad)), one(rad))
      pz = vifelse(alive, pz + (sqrt_rad - Pc)/P0c, pz)  

      v[i,PZI] = vifelse(alive, v[i,PZI] + dE/P0c, v[i,PZI])
      to_momentum_coords!(i, coords, mass, P0c, pz)

      #

      if !isnothing(coords.q)
        rotate_spin_field!(i, coords, a, g_bend, mass/P0c, a_potential, a_potential, e_field, b_field, 0.5)
      end
    end

    function potential_kick!(i, coords::Coords, 
                                a, q_voltage, rf_omega, t_phi0, t_ref, mass, P0c, K)
      v = coords.v
      alive = (coords.state[i] == STATE_ALIVE)

      pz = v[i,PZI]
      Pc = (1 + pz) * P0c
      m_over_pc = mass / ((1 + v[i,PZI]) * P0c)
      beta = 1 / sqrt(1 + m_over_pc * m_over_pc)
      t = t_phi0 + t_ref - v[i,ZI] / (beta * c_light(eltype(coords.v)))
      dE = q_voltage * cos(rf_omega * t)

      # 

      if !isnothing(coords.q)
        e_field = (0, 0, dE * c_light(eltype(coords.v)) / P0c)
        b_field = (0, 0, 0)
        ax = -K*v[i,YI]/2
        ay = K*v[i,XI]/2
        g_bend = 0
        rotate_spin_field!(i, coords, a, g_bend, mass/P0c, ax, ay, e_field, b_field, 0.5)
      end

      # Energy kick

      to_energy_coords!(i, coords, beta)
      rad = dE*dE + 2*sqrt(Pc*Pc + mass^2) * dE + Pc*Pc
      coords.state[i] = vifelse(rad < 0, STATE_LOST_PZ, coords.state[i])
      alive = (coords.state[i] == STATE_ALIVE)
      sqrt_rad = vifelse(alive, sqrt(abs(rad)), one(rad))
      pz = vifelse(alive, pz + (sqrt_rad - Pc)/P0c, pz)  

      v[i,PZI] = vifelse(alive, v[i,PZI] + dE/P0c, v[i,PZI])
      to_momentum_coords!(i, coords, mass, P0c, pz)

      #

      if !isnothing(coords.q)
        rotate_spin_field!(i, coords, a, g_bend, mass/P0c, ax, ay, e_field, b_field, 0.5)
      end
    end

    end

    function initial(K; axis=false, moving=false, translated=false)
        v = zeros(16,6)
        for i in 1:16
            x = axis ? 0.0 : 0.006*(i-8.5)
            y = axis ? 0.0 : 0.013sin(i)
            if translated
                x += 0.021; y -= 0.017
            end
            v[i,1] = x; v[i,3] = y
            v[i,2] = -K*y/2 + (moving ? 0.014cos(i) : 0.0)
            v[i,4] = K*x/2 + (moving ? -0.009sin(i) : 0.0)
        end
        return v
    end

    # RF at the accelerating crest, 91 MHz, dE_ref = 0, radiation off; gain = 0 selects the pure
    # Exact-solenoid control.
    function cavity(sp, pc, K, L, gain)
        if gain == 0
            return Beamline([Solenoid(L=L, Ksol=K, tracking_method=Exact(fringe_at=Fringe.NoEnd))], species_ref=sp, pc_ref=pc)
        end
        # Accelerating phase adds pi for a negative species, making positive
        # voltage accelerate either sign of charge.
        rf = RFCavity(L=L, Ksol=K, voltage=gain/abs(chargeof(sp)), rf_frequency=9.1e7,
            zero_phase=PhaseRef.Accelerating, tracking_method=SaganCavity(L_active=0.0))
        # A zero-increment downstream Beamline keeps the reference fixed.
        br = Branch([Beamline([Marker()], species_ref=sp, pc_ref=pc), Beamline([rf], dE_ref=0.0)])
        return br.beamlines[2]
    end

    function make_bunch(v, sp, pc)
        return Bunch(v=copy(v), spin=true, species=sp, p_over_q_ref=pc/(chargeof(sp)*C_LIGHT))
    end

    function longitudinal_oracle(sp, pc, K, L, gain)
        return setprecision(256) do
            p = BigFloat(pc); m = BigFloat(massof(sp))
            uf = sqrt((sqrt(p*p + m*m) + BigFloat(gain))^2 - m*m)/p
            angle = -(1+BigFloat(gyromagnetic_anomaly(sp)))*BigFloat(K)*BigFloat(L)/2*(1+inv(uf))
            Float64(uf-1), Float64.([cos(angle/2), 0, 0, sin(angle/2)])
        end
    end

    function main()
        count = 0; max_orbit = 0.0; max_dp = 0.0; max_norm = 0.0; max_control = 0.0
        max_transverse = 0.0
        for species in ("proton", "electron"), bg in (0.6, 2.0, 8.0), K in (-1.4, 0.0, 1.4),
            fraction in (-2e-6, 0.0, 2e-6, 2e-3), L in (0.43, 1e-8), simd in (false, true)
            sp = Species(species); m = massof(sp); pc = bg*m; gain = fraction*m
            v = initial(K)
            b = make_bunch(v, sp, pc)
            track!(b, cavity(sp, pc, K, L, gain); use_explicit_SIMD=simd)
            dp, q = longitudinal_oracle(sp, pc, K, L, gain)
            orbit = maximum(abs, b.v[:,1:4]-v[:,1:4])
            dp_error = maximum(abs, b.v[:,6].-dp)
            norm_error = maximum(abs, sum(abs2,b.q; dims=2).-1)
            transverse = maximum(2hypot(b.q[i,2],b.q[i,3])*hypot(b.q[i,1],b.q[i,4]) for i in 1:16)
            q_error = maximum(abs, b.q.-permutedims(q))
            @assert all(b.state .== BeamTracking.STATE_ALIVE)
            @assert orbit < 2e-16
            @assert dp_error < 2e-15
            @assert norm_error < 5e-15
            if K == 0 || gain == 0
                @assert q_error < 3e-15
                max_control = max(max_control, q_error)
            else
                @assert transverse > 1e-12
            end
            max_orbit = max(max_orbit,orbit); max_dp = max(max_dp,dp_error)
            max_norm = max(max_norm,norm_error); max_transverse = max(max_transverse,transverse)
            count += 1
            if bg == 2 && K == 1.4 && fraction in (-2e-6,2e-6,2e-3) && !simd
                @printf("longitudinal species=%s bg=%.1f K=%.1f L=%.3g gain=%.17g qx16=%.17g qy16=%.17g max_transverse_spin=%.17g q_error=%.17g\n", species,bg,K,L,gain,b.q[16,2],b.q[16,3],transverse,q_error)
            end
        end
        @printf("longitudinal sweep cases=%d max_orbit=%.17g max_delta_error=%.17g max_qnorm_error=%.17g zero_field_or_gain_qerror=%.17g max_transverse_spin=%.17g\n",count,max_orbit,max_dp,max_norm,max_control,max_transverse)

        for species in ("proton", "electron")
            sp = Species(species); m = massof(sp); pc = 2m; gain = 2e-3m; K = 1.4; L = 0.43
            bl = cavity(sp,pc,K,L,gain)
            v0 = initial(K; axis=true, moving=true)
            v1 = initial(K; axis=true, moving=true, translated=true)
            b0 = make_bunch(v0,sp,pc); b1 = make_bunch(v1,sp,pc)
            track!(b0,bl;use_explicit_SIMD=false); track!(b1,bl;use_explicit_SIMD=false)
            orbit_difference = maximum(abs, (b1.v-b0.v)-(v1-v0))
            spin_difference = maximum(abs,b1.q-b0.q)
            @assert all(b0.state .== BeamTracking.STATE_ALIVE) && all(b1.state .== BeamTracking.STATE_ALIVE)
            @assert orbit_difference < 2e-16
            @assert spin_difference > 1e-7
            @printf("translation species=%s dx=0.021 dy=-0.017 orbit_symmetry_error=%.17g spin_symmetry_error=%.17g\n",species,orbit_difference,spin_difference)

            # An impulse alone removes every magnetic transport approximation.
            stationary = initial(K)
            half_b = make_bunch(stationary,sp,pc)
            fixed_b = make_bunch(stationary,sp,pc)
            for i in 1:16
                ControlKicks.half_kick!(i,half_b.coords,gyromagnetic_anomaly(sp),gain,2pi*9.1e7,0.,0.,m,pc)
                ControlKicks.potential_kick!(i,fixed_b.coords,gyromagnetic_anomaly(sp),gain,2pi*9.1e7,0.,0.,m,pc,K)
            end
            half_error = maximum(abs,half_b.q[:,2:3])
            fixed_error = maximum(abs,fixed_b.q.-permutedims([1.,0.,0.,0.]))
            @assert half_error > 1e-7
            @assert fixed_error == 0
            @assert half_b.v == fixed_b.v
            @printf("COL11_half_only species=%s max_transverse_q=%.17g with_potential_qerror=%.17g orbital_difference=0\n",species,half_error,fixed_error)

            # Nonzero mechanical momentum checks that the supplied potential does
            # more than force a special zero-velocity branch to return zero.
            local0 = make_bunch(v0,sp,pc); local1 = make_bunch(v1,sp,pc)
            for i in 1:16
                ControlKicks.potential_kick!(i,local0.coords,gyromagnetic_anomaly(sp),gain,2pi*9.1e7,0.,0.,m,pc,K)
                ControlKicks.potential_kick!(i,local1.coords,gyromagnetic_anomaly(sp),gain,2pi*9.1e7,0.,0.,m,pc,K)
            end
            fixed_translation = maximum(abs,local0.q-local1.q)
            @assert fixed_translation < 2e-18
            @printf("corrected_impulse_translation species=%s spin_symmetry_error=%.17g nonzero_q=%.17g\n",species,fixed_translation,maximum(abs,local0.q[:,2:3]))
        end
        println("expected: longitudinal electric spin rotation=0; uniform-field translation preserves spin")
        println("DEFECT REPRODUCED")
    end
    main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | BeamTracking main (`d8e45fdc25db`), source, Julia 1.13.2, CPU, Float64, serial |

