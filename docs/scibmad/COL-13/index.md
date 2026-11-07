---
title: "COL-13: Automatic Float32 mesh bounds invalidate CIC cells for input particles"
hide:
  - toc
---

# COL-13 — Automatic Float32 mesh bounds invalidate CIC cells for input particles

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on SpaceCharge release [v1.3.0](https://github.com/bmad-sim/SpaceCharge.jl/tree/v1.3.0) (`ae30463`) |
| Source | [`SpaceCharge.jl/src/mesh.jl:155-178`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/mesh.jl#L155-L178)<br>[`src/deposition.jl:38-84`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/deposition.jl#L38-L84)<br>[`mesh.jl:99-117`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/mesh.jl#L99-L117)<br>[`test/test_mesh.jl:61-69`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/test/test_mesh.jl#L61-L69) |

The automatic `Mesh3D` constructor with Float32 precision can create bounds that leave its own input particles outside valid cloud-in-cell (CIC) deposition cells. The constructor promises padded bounds suitable for deposition, but exact rational arithmetic shows that Float32 rounding removes the required margin. For input horizontal positions 0 and 1.000 m on a 65×9×9 mesh, deposition addresses nodes 65 and 66 although only 65 exist, raising `BoundsError` instead of depositing the total 2.000 pC; a 17-node grid also fails even though its upper bound exceeds 1 m. Automatic Float64 bounds and sufficiently padded manual Float32 bounds deposit successfully.

Run `julia COL-13.jl`.

??? example "Reproducer: COL-13.jl"

    [Download COL-13.jl](COL-13.jl)

    ```julia
    # COL-13: Automatic Float32 mesh bounds invalidate CIC cells for input particles
    #
    # The automatic `Mesh3D` constructor with Float32 precision can create bounds that leave its own
    # input particles outside valid cloud-in-cell (CIC) deposition cells. The constructor promises
    # padded bounds suitable for deposition, but exact rational arithmetic shows that Float32 rounding
    # removes the required margin. For input horizontal positions 0 and 1.000 m on a 65×9×9 mesh,
    # deposition addresses nodes 65 and 66 although only 65 exist, raising `BoundsError` instead of
    # depositing the total 2.000 pC; a 17-node grid also fails even though its upper bound exceeds 1 m.
    # Automatic Float64 bounds and sufficiently padded manual Float32 bounds deposit successfully.
    #
    # The script builds automatic Float32 meshes of 9, 17, 33, 65 and 129 × 9 × 9 nodes around two
    # particles at x = 0 and 1 m, deposits their charges, and checks each outcome against the cell index
    # predicted from the mesh bounds. Controls deposit the same particles on an automatic Float64 mesh
    # and on a manual Float32 mesh padded by 10⁻⁵ m, and exact rational arithmetic shows that the
    # 65-node padding is smaller than half a Float32 unit in the last place at 1.
    using SpaceCharge, Printf
    using LinearAlgebra
    BLAS.set_num_threads(1)
    SpaceCharge.FFTW.set_num_threads(1)

    # Nondegenerate extents in every direction; only x has the fine grid.
    # Reverse z ordering so this is not just two opposite unit-cube corners.
    const X = Float32[0,1]
    const Y = Float32[-0.25,0.25]
    const Z = Float32[0.5,-0.5]
    const Q = Float32[1.25e-12,0.75e-12]
    const TOTAL = sum(Float64,Q)

    function deposit_check(mesh,label)
        try
            deposit!(mesh,X,Y,Z,Q)
            charge = sum(Float64,mesh.rho)
            error = abs(charge-TOTAL)
            # Only roundoff from the eight Float32 CIC weights is allowed.
            @assert error <= 4*eps(Float32)*TOTAL
            @printf("%s deposit=ok charge=%.17g expected=%.17g absolute_error=%.3g\n",label,charge,TOTAL,error)
            return true
        catch err
            err isa BoundsError || rethrow()
            println(label," deposit=BoundsError indices=",err.i," array_size=",size(err.a))
            return false
        end
    end

    function main()
        for nx in (9,17,33,65,129)
            mesh = Mesh3D((nx,9,9),X,Y,Z;T=Float32)
            ux = (X[2]-mesh.min_bounds[1])/mesh.delta[1]
            lower = floor(Int,ux)+1
            safe = lower+1 <= nx
            @printf("auto Float32 nx=%d min=%.17g max=%.17g delta=%.17g u=%.17g lower_1based=%d safe=%s\n",
                    nx,mesh.min_bounds[1],mesh.max_bounds[1],mesh.delta[1],ux,lower,safe)
            ok = deposit_check(mesh,"auto Float32 nx=$nx")
            @assert ok == safe
            nx == 65 && @assert !ok
        end

        mesh64 = Mesh3D((65,9,9),X,Y,Z;T=Float64)
        @assert deposit_check(mesh64,"auto Float64")
        lo = (-1e-5,-0.25-1e-5,-0.5-1e-5)
        hi = (1+1e-5,0.25+1e-5,0.5+1e-5)
        manual = Mesh3D((65,9,9),lo,hi;T=Float32)
        @assert deposit_check(manual,"manual Float32 representable padding")

        # An exact integer lattice-index oracle, independent of deposit!'s floor logic.
        # Padding is 10^-6 * (1/64); it is below half an ulp at x=1.
        pad = big(1)//(big(64)*10^6)
        half_ulp_at_one = big(1)//big(2)^24
        println("nx=65 exact padding=",Float64(pad)," half_ulp_above_1=",Float64(half_ulp_at_one))
        @assert pad < half_ulp_at_one
        mesh = Mesh3D((65,9,9),X,Y,Z;T=Float32)
        @assert mesh.max_bounds[1] == 1f0
        @assert mesh.min_bounds[1] < 0f0  # Lower padding survives conversion.
        @assert mesh.delta[1] == 1f0/64
        @assert (X[2]-mesh.min_bounds[1]) == 1f0  # But subtraction rounds it away.
        @assert (X[2]-mesh.min_bounds[1])/mesh.delta[1] == 64f0
        println("nx=65: upper point requires nodes 65 and 66; grid has 65 nodes")
        println("DEFECT REPRODUCED")
    end
    main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float32 and Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | SpaceCharge 1.3.0 (`ae3046330572`), source, Julia 1.13.2, CPU, Float32 and Float64, serial |

