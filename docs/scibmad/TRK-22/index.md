---
title: "TRK-22: Neutron g_spin conflicts with its own-mass normalization contract"
hide:
  - toc
---

# TRK-22 — Neutron g_spin conflicts with its own-mass normalization contract

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on AtomicAndPhysicalConstants main (`750a9f2`) |
| Discussed in | [[gh:https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/pull/306]] |
| Source | [src/constants.jl:177](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/src/constants.jl#L177)<br>[src/species_data.jl:26](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/src/species_data.jl#L26)<br>[src/species_data.jl:42](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/src/species_data.jl#L42)<br>[src/constructors.jl:11–43](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/src/constructors.jl#L11-L43)<br>[src/functions.jl:164–167](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/src/functions.jl#L164-L167)<br>[test/values_test.jl:34](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/test/values_test.jl#L34)<br>[src/functions.jl:142–150](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/src/functions.jl#L142-L150)<br>[docs/src/constants.md:83–113](https://github.com/bmad-sim/AtomicAndPhysicalConstants.jl/blob/4095d795a50aafd63687f66f868bb4adcb034a08/docs/src/constants.md#L83-L113) |

The neutron and antineutron `g_spin` values are normalized to the nuclear magneton, which uses the proton mass, although the accessor promises normalization to each particle's own mass. Combining the returned value with the documented formula therefore disagrees with the tabulated magnetic moment, and converting by the neutron-to-proton mass ratio restores agreement. With CODATA 2022 active, neutron `g_spin` with `signed=true` is −3.826 instead of the own-mass value −3.831, implying a gyromagnetic ratio of −1.830 × 10⁸ instead of −1.832 × 10⁸ s⁻¹ T⁻¹, a 0.1377% magnitude error. Antineutron signs reverse with the same fractional error; charged species agree.

??? example "Reproducer: TRK-22.jl"

    [Download TRK-22.jl](TRK-22.jl)

    ```julia
    # TRK-22: Neutron g_spin conflicts with its own-mass normalization contract
    #
    # The neutron and antineutron `g_spin` values are normalized to the nuclear magneton, which uses the
    # proton mass, although the accessor promises normalization to each particle's own mass. Combining
    # the returned value with the documented formula therefore disagrees with the tabulated magnetic
    # moment, and converting by the neutron-to-proton mass ratio restores agreement. With CODATA 2022
    # active, neutron `g_spin` with `signed=true` is −3.826 instead of the own-mass value −3.831,
    # implying a gyromagnetic ratio of −1.830 × 10⁸ instead of −1.832 × 10⁸ s⁻¹ T⁻¹, a 0.1377% magnitude
    # error. Antineutron signs reverse with the same fractional error; charged species agree.
    #
    # With CODATA 2022 active, the script compares, for the neutron, the antineutron and twelve charged
    # controls, the gyromagnetic ratio g·e/(2m) from `g_spin` and the particle's own mass m with μ/(Sℏ)
    # from the stored magnetic moment μ and spin S, in SI units. It checks that rescaling the neutron g
    # by the neutron-to-proton mass ratio restores agreement, and repeats the comparison as a
    # magnetic-moment identity in eV. The charged controls must agree.

    using AtomicAndPhysicalConstants

    # Independent SI constants and the measured gyromagnetic ratio from the 2022
    # CODATA table, https://physics.nist.gov/cuu/pdf/all.pdf, pages 1 and 4.
    const e_si = 1.602176634e-19
    const c_si = 299792458.0
    const h_si = 6.62607015e-34
    const hbar_si = h_si/(2pi)
    const gamma_n_codata = 1.83247174e8
    const gamma_n_uncertainty = 43.0

    # g_spin(neutron) returns CODATA's value normalized to the nuclear magneton e hbar/(2 m_p), while
    # the accessor docstring and manual promise mu = g (e/2m) S with the particle's own mass.
    function main()
        println("active_release=",RELEASE_YEAR)
        @assert RELEASE_YEAR == 2022 || RELEASE_YEAR == "2022"
        for name in ("neutron", "anti-neutron", "electron", "positron", "muon", "anti-muon",
                "proton", "anti-proton", "deuteron", "anti-deuteron", "triton", "anti-triton", "helion", "anti-helion")
            sp = Species(name)
            m_si = massof(sp)*e_si/c_si^2
            moment_si = momentof(sp)*e_si
            gamma_from_g = g_spin(sp;signed=true)*e_si/(2m_si)
            gamma_from_moment = moment_si/(spinof(sp)*hbar_si)
            residual = gamma_from_g/gamma_from_moment-1
            println("SPECIES ",name," g_signed=",g_spin(sp;signed=true)," g_abs=",g_spin(sp),
                " mass_kg=",m_si," moment_J_per_T=",moment_si," gamma_g=",gamma_from_g,
                " gamma_moment=",gamma_from_moment," relative_error=",residual)
            @assert g_spin(sp) == abs(g_spin(sp;signed=true))
            if occursin("neutron",name)
                @assert abs(residual) > 1e-3
                @assert abs(abs(gamma_from_moment)-gamma_n_codata) < gamma_n_uncertainty
                rescaled_gamma = gamma_from_g*M_NEUTRON/M_PROTON
                println("NEUTRON rescaled_gamma=",rescaled_gamma,
                    " discrepancy_in_CODATA_sigmas=",abs(abs(gamma_from_g)-gamma_n_codata)/gamma_n_uncertainty,
                    " required_own_mass_g=",gamma_from_moment*2m_si/e_si,
                    " nuclear_convention_g=",moment_si/(0.5*e_si*hbar_si/(2*(M_PROTON*e_si/c_si^2))))
                @assert abs(abs(rescaled_gamma)-gamma_n_codata) < gamma_n_uncertainty
                @assert isnan(gyromagnetic_anomaly(sp))
            else
                @assert abs(residual) < 2e-9
            end
        end
        sp=Species("neutron")
        gamma_g=abs(g_spin(sp;signed=true))*e_si/(2massof(sp)*e_si/c_si^2)
        for B in (0.02,1.0,7.0)
            println("DERIVED_PRECESSION B_T=",B," frequency_from_g_Hz=",gamma_g*B/(2pi),
                " frequency_from_CODATA_Hz=",gamma_n_codata*B/(2pi))
        end
        # Inspect immutable release records without changing active preferences.
        for release in (CODATA2002,CODATA2006,CODATA2010,CODATA2014,CODATA2018,CODATA2022)
            println("RAW_RELEASE G_NEUTRON=",release.G_NEUTRON," mn_over_mp=",release.M_NEUTRON/release.M_PROTON)
        end
        println("CONFIRMED neutron own-mass contract mismatch; no neutral tracking asserted")
    end
    main()

    module MomentIdentityCheck
    using AtomicAndPhysicalConstants, Printf
    const APC=AtomicAndPhysicalConstants
    println("CODATA release=",APC.RELEASE_YEAR)
    println("Independent magnetic-moment identity: mu = g * I * hbar * c^2 / (2*m), using eV units.")
    for name in ("electron","positron","muon","anti-muon","proton","anti-proton",
                 "deuteron","anti-deuteron","triton","anti-triton","helion","anti-helion",
                 "neutron","anti-neutron")
        s=Species(name)
        g=g_spin(s;signed=true)
        implied=g*spinof(s)*APC.H_BAR*C_LIGHT^2/(2massof(s))
        physical=momentof(s)
        inferred=2massof(s)*physical/(spinof(s)*APC.H_BAR*C_LIGHT^2)
        @printf("%-14s g_actual=% .17g g_from_moment=% .17g mu_actual=% .17g mu_from_g=% .17g relative_moment_error=% .17g\n",
            name,g,inferred,physical,implied,implied/physical-1)
        if occursin("neutron",name)
            @assert abs(implied/physical-1)>0.001
            @assert isnan(gyromagnetic_anomaly(s))
        else
            @assert abs(implied/physical-1)<2e-8
        end
    end
    rescaled=APC.G_NEUTRON*APC.M_NEUTRON/APC.M_PROTON
    println("NIST nuclear-magneton conversion g_expected=",rescaled," g_actual=",g_spin(Species("neutron");signed=true))
    println("converted neutron moment=",rescaled*0.5*APC.H_BAR*C_LIGHT^2/(2APC.M_NEUTRON)," tabulated=",APC.MU_NEUTRON)
    println("mass ratio=",APC.M_NEUTRON/APC.M_PROTON," predicted relative error=",APC.M_PROTON/APC.M_NEUTRON-1)
    println("NEUTRON NORMALIZATION DEFECT REPRODUCED; neutral-particle tracking impact is unclaimed")
    end

    println("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | AtomicAndPhysicalConstants main (`4095d795a50a`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | AtomicAndPhysicalConstants main (`4095d795a50a`), source, Julia 1.13.2, CPU, Float64, serial |

