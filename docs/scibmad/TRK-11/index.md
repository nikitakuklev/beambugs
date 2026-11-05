---
title: "TRK-11: Sagan physics guide: edge kicks ×2, contradictory signs, ponderomotive H sign"
hide:
  - toc
---

# TRK-11 — Sagan physics guide: edge kicks ×2, contradictory signs, ponderomotive H sign

| | |
|---|---|
| Code | [SciBmad](../index.md) · Tracking |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Source | [docs/src/sagancavity-physics.md:59–69](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-physics.md#L59-L69)<br>[docs/src/sagancavity-physics.md:73–81](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-physics.md#L73-L81)<br>[docs/src/sagancavity-physics.md:83–84](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-physics.md#L83-L84)<br>[docs/src/sagancavity-physics.md:159–164](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-physics.md#L159-L164)<br>[docs/src/sagancavity-physics.md:170–181](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/sagancavity-physics.md#L170-L181)<br>[src/kernels/sagan_cavity.jl:252–279](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/sagan_cavity.jl#L252-L279)<br>[bmad/low_level/track_a_lcavity.f90:265–308](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/low_level/track_a_lcavity.f90#L265-L308)<br>[src/kernels/sagan_cavity.jl:222–231](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/kernels/sagan_cavity.jl#L222-L231) |

The `SaganCavity` physics guide doubles the edge kicks, gives Hamiltonians whose signs contradict its own listed kicks, and calls the entrance of an accelerating cavity defocusing. Hamilton's equations, the implemented kernels and Bmad's edge model instead give half the guide's edge kick, a focusing entrance and a positive ponderomotive Hamiltonian, which generates the kernel's focusing kick. For a proton with reference momentum p₀c = 2mc², twice its rest energy, charge-weighted accelerating gradient qG = 1.000 MeV/m and horizontal offset 1.000 mm, the guide's entrance kick in normalized horizontal momentum is −5.329 × 10⁻⁷ instead of −2.664 × 10⁻⁷, and its edge Hamiltonian implies +5.329 × 10⁻⁷. The kernels agree with Bmad.

??? example "Reproducer: TRK-11.jl"

    [Download TRK-11.jl](TRK-11.jl)

    ```julia
    # TRK-11: Sagan physics guide: edge kicks ×2, contradictory signs, ponderomotive H sign
    #
    # The `SaganCavity` physics guide doubles the edge kicks, gives Hamiltonians whose signs contradict
    # its own listed kicks, and calls the entrance of an accelerating cavity defocusing. Hamilton's
    # equations, the implemented kernels and Bmad's edge model instead give half the guide's edge kick,
    # a focusing entrance and a positive ponderomotive Hamiltonian, which generates the kernel's
    # focusing kick. For a proton with reference momentum p₀c = 2mc², twice its rest energy,
    # charge-weighted accelerating gradient qG = 1.000 MeV/m and horizontal offset 1.000 mm, the guide's
    # entrance kick in normalized horizontal momentum is −5.329 × 10⁻⁷ instead of −2.664 × 10⁻⁷, and its
    # edge Hamiltonian implies +5.329 × 10⁻⁷. The kernels agree with Bmad.
    #
    # The script calls the `SaganCavity` edge and ponderomotive kernels directly for a proton with p₀c =
    # 2mc² at horizontal offset 1 mm, with qG = 1 MeV/m on crest, and evaluates both the guide's kick
    # equations and the kicks its Hamiltonians imply through Hamilton's equations. It checks that the
    # guide's entrance kick is twice the kernel's, that its edge Hamiltonian implies the opposite sign,
    # that the kernel's entrance kick focuses, and that the guide's ponderomotive Hamiltonian implies
    # the opposite of its own focusing kick.
    using SciBmad
    const BT = SciBmad.BeamTracking

    sp = Species("proton"); M = massof(sp); P0c = 2M
    qG = 1e6; x = 1e-3; phi = 0.0                 # crest: cos(phi) = 1
    b = Bunch([x 0.0 0.0 0.0 0.0 0.0], species = sp, p_over_q_ref = P0c / C_LIGHT)
    BT.sagan_cavity_fringe!(1, b.coords, 0.0, qG, 2pi * 1e8, phi, 0.0, M, P0c, +1)
    dpx_code = b.v[1, 2]
    # Guide, entrance: H_f = -(q G cos(phi)/(2 P0c)) (x^2+y^2) and dpx = -(q G cos(phi)/P0c) x. The
    # kernel and Fortran Bmad apply half that kick, focusing at the entrance on an accelerating crest.
    dpx_guide_kick = -qG * cos(phi) / P0c * x          # guide kick equation, entrance (upper sign)
    dpx_guide_H = +2 * qG * cos(phi) / (2P0c) * x      # -dH_f/dx with the guide's entrance H_f
    println("entrance dpx: kernel = ", dpx_code)
    println("              guide kick equation = ", dpx_guide_kick, "  (ratio guide/kernel = ", dpx_guide_kick / dpx_code, ")")
    println("              guide Hamiltonian -dH/dx = ", dpx_guide_H, "  (sign opposite to the guide's own kick)")
    println("              kernel kick on accelerating crest is ", dpx_code * x < 0 ? "focusing" : "defocusing",
            "; guide prose says entrance is defocusing")

    b2 = Bunch([x 0.0 0.0 0.0 0.0 0.0], species = sp, p_over_q_ref = P0c / C_LIGHT)
    Lc = 0.1
    BT.rf_pondermotive_kick!(1, b2.coords, qG, 0.0, M, P0c, Lc)
    # Guide: H_p = -A (x^2+y^2) with listed kick dpx = -2A x, but Hamilton's equations give dpx =
    # -dH_p/dx = +2A x.
    A = qG^2 * Lc / (16 * P0c^2)                       # guide's H_p = -A (x^2+y^2) (kappa_n/N_s -> 1, pz = 0)
    println("ponderomotive dpx: kernel = ", b2.v[1, 2], "; guide kick = ", -2A * x, "; -dH_p/dx from guide H_p = ", 2A * x)
    bad = abs(dpx_guide_kick / dpx_code - 2) < 1e-12 && dpx_code * x < 0 && sign(dpx_guide_H) != sign(dpx_guide_kick) &&
          abs(b2.v[1, 2] + 2A * x) < 1e-20
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release clones (BeamTracking 0.8.7 65e343f9afda), scibmad-ref environment, Julia 1.13.2, CPU, Float64 |

