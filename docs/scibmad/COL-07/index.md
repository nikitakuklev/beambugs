---
title: "COL-07: Degenerate automatic z bounds misplace the cathode image"
hide:
  - toc
---

# COL-07 — Degenerate automatic z bounds misplace the cathode image

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SpaceCharge release [v1.3.0](https://github.com/bmad-sim/SpaceCharge.jl/tree/v1.3.0) (`ae30463`) |
| Source | [`SpaceCharge.jl/src/mesh.jl:142-173`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/mesh.jl#L142-L173)<br>[`SpaceCharge.jl/src/solvers/free_space.jl:25-31`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/solvers/free_space.jl#L25-L31)<br>[`SpaceCharge.jl/src/fft/transforms.jl:26,38`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/fft/transforms.jl#L26-L38)<br>[`free_space.jl:12`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/solvers/free_space.jl#L12)<br>[`mesh.jl:233-244`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/mesh.jl#L233-L244)<br>[`mesh.jl:168-173`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/mesh.jl#L168-L173)<br>[`SpaceCharge.jl/examples/cathode_solver_example.jl:8-21`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/examples/cathode_solver_example.jl#L8-L21) |

When every particle has the same longitudinal position z, the automatic `Mesh3D` bounds misplace the mirror image that `solve!(at_cathode=true)` uses to model a conducting cathode plane. Electrostatics puts an equal and opposite image charge at the mirror point across the cathode plane, and a mesh with consistent manual bounds reproduces that analytic field. For 1.000 nC at 100.0 µm from the cathode on a 32³ mesh, the computed axial field is 1.401 times the correct value, matching an image 31.00 µm closer than it should be. The shipped cathode example falls in this case, with a field 0.3106% high; inputs with distinct z values and consistent manual bounds avoid this displacement.

Run `julia COL-07.jl`.

??? example "Reproducer: COL-07.jl"

    [Download COL-07.jl](COL-07.jl)

    ```julia
    # COL-07: Degenerate automatic z bounds misplace the cathode image
    #
    # When every particle has the same longitudinal position z, the automatic `Mesh3D` bounds misplace
    # the mirror image that `solve!(at_cathode=true)` uses to model a conducting cathode plane.
    # Electrostatics puts an equal and opposite image charge at the mirror point across the cathode
    # plane, and a mesh with consistent manual bounds reproduces that analytic field. For 1.000 nC at
    # 100.0 µm from the cathode on a 32³ mesh, the computed axial field is 1.401 times the correct
    # value, matching an image 31.00 µm closer than it should be. The shipped cathode example falls in
    # this case, with a field 0.3106% high; inputs with distinct z values and consistent manual bounds
    # avoid this displacement.
    #
    # The script deposits 1 nC on the axis at z₀ = 100 µm and at 1 cm on a 32³ mesh with automatic
    # bounds from that single particle, solves with `at_cathode=true`, and compares the axial field with
    # the point-image value −Q/(4πε₀(2z₀)²) and with an image moved closer by (n_z − 1)·Δz = 31 µm. As
    # controls, it repeats the 100 µm case on a mesh with consistent manual bounds and on automatic
    # bounds built from two distinct z values.
    using SpaceCharge, LinearAlgebra
    const SC = SpaceCharge

    Q = 1e-9; n = 32
    for z0 in (1e-4, 0.01)
        m = Mesh3D((n, n, n), [0.0], [0.0], [z0])
        deposit!(m, [0.0], [0.0], [z0], [Q])
        solve!(m; at_cathode = true, fft_flags = SC.FFTW.ESTIMATE)
        Ex, Ey, Ez = first.(interpolate_field(m, [0.0], [0.0], [z0]))
        dz = m.delta[3]
        # The image of Q at z0 is -Q at -z0, so on axis Ez(z0) = -Q/(4 pi eps0 (2 z0)^2). The automatic
        # constructor keeps zmin = zmax = z0 but replaces the zero spacing by 1e-6, so max_bounds is
        # not zmin + (nz-1)*dz. The image offset zmin + zmax assumes it is, which moves the image to
        # distance 2 z0 - (nz-1)*dz. The shipped cathode_solver_example.jl uses exactly this setup.
        Ecorrect = -SC.FPEI * Q / (2z0)^2
        Eshift = -SC.FPEI * Q / (2z0 - (n - 1) * dz)^2
        println("z0 = $z0 m: min/max z bound = ", m.min_bounds[3], " / ", m.max_bounds[3], ", dz = ", dz)
        println("  Ez solver = ", Ez, " V/m")
        println("  Ez image at 2 z0 (point limit) = ", Ecorrect, "   solver/correct = ", Ez / Ecorrect)
        println("  Ez image at 2 z0 - (nz-1) dz   = ", Eshift, "   solver/shifted = ", Ez / Eshift)
    end
    # Independent recheck: same charge on a mesh with consistent manual bounds, and with
    # automatic bounds from two distinct z values (nondegenerate spacing).
    dz1 = 1e-6; z1 = 1e-4
    mm = Mesh3D((n, n, n), (-16dz1, -16dz1, z1), (15dz1, 15dz1, z1 + (n - 1) * dz1))
    deposit!(mm, [0.0], [0.0], [z1], [Q]); solve!(mm; at_cathode = true, fft_flags = SC.FFTW.ESTIMATE)
    println("recheck: manual bounds   Ez/correct = ", interpolate_field(mm, [0.0], [0.0], [z1])[3][1] / (-SC.FPEI * Q / (2z1)^2))
    ma = Mesh3D((n, n, n), [0.0, 1e-6], [0.0, 1e-6], [z1, z1 + 31e-6])
    deposit!(ma, [0.0], [0.0], [z1], [Q]); solve!(ma; at_cathode = true, fft_flags = SC.FFTW.ESTIMATE)
    println("recheck: automatic bounds, nondegenerate z: Ez/correct = ", interpolate_field(ma, [0.0], [0.0], [z1])[3][1] / (-SC.FPEI * Q / (2z1)^2))
    m = Mesh3D((n, n, n), [0.0], [0.0], [1e-4]); deposit!(m, [0.0], [0.0], [1e-4], [Q])
    solve!(m; at_cathode = true, fft_flags = SC.FFTW.ESTIMATE)
    Ez = interpolate_field(m, [0.0], [0.0], [1e-4])[3][1]
    println(abs(Ez / (-SC.FPEI * Q / (2e-4)^2) - 1) > 0.1 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float64, serial |

