---
title: "COL-18: Half-cell offsets in multiple directions make the documented free-space solver nonfinite"
hide:
  - toc
---

# COL-18 — Half-cell offsets in multiple directions make the documented free-space solver nonfinite

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on SpaceCharge release [v1.3.0](https://github.com/bmad-sim/SpaceCharge.jl/tree/v1.3.0) (`ae30463`) |
| Source | [`src/solvers/free_space.jl:20–55`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/solvers/free_space.jl#L20-L55)<br>[`src/green_functions.jl:69–95`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L69-L95)<br>[`src/green_functions.jl:24–37`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L24-L37)<br>[`41–66`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L41-L66)<br>[`src/green_functions.jl:160–175`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/green_functions.jl#L160-L175)<br>[`test/test_solver_reference.jl:15–22`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/test/test_solver_reference.jl#L15-L22)<br>[`src/SpaceCharge.jl:19`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/SpaceCharge.jl#L19)<br>[`src/deposition.jl:59–84`](https://github.com/bmad-sim/SpaceCharge.jl/blob/ae30463305729aec2f04d4687ffd547b94750cb2/src/deposition.jl#L59-L84) |

Calling the documented free-space solver with half-cell offsets in two coordinate directions returns nonfinite electric fields for a finite charge distribution. On a 4³ mesh with 0.9766 mm spacing and a 1 pC source, 128 of 192 field components become nonfinite; shifting all three directions makes all 192 nonfinite. Independent surface integration of the Coulomb field gives (3559, 1166, 0) V/m at a sample where the solver returns NaN in both transverse components. Full and pruned FFTs, with and without Green-function caching, exhibit the failure. The exported free-space and cathode solves use zero transverse offsets and avoid this trigger.

Run `python COL-18.py` with [_scibmad.py](../helpers/_scibmad.py) in the same directory.

??? example "Reproducer: COL-18.py"

    [Download COL-18.py](COL-18.py)

    ```python
    """COL-18: Half-cell offsets in multiple directions make the documented free-space solver nonfinite

    Calling the documented free-space solver with half-cell offsets in two coordinate directions returns nonfinite electric fields for a finite charge distribution. On a 4³ mesh with 0.9766 mm spacing and a 1 pC source, 128 of 192 field components become nonfinite; shifting all three directions makes all 192 nonfinite. Independent surface integration of the Coulomb field gives (3559, 1166, 0) V/m at a sample where the solver returns NaN in both transverse components. Full and pruned FFTs, with and without Green-function caching, exhibit the failure. The exported free-space and cathode solves use zero transverse offsets and avoid this trigger.

    Setup: `SCIBMAD_JULIA` set to a Julia launcher, an executable that starts Julia in the SciBmad environment; the Python script runs its embedded Julia program through it.

    The script deposits one fixed node charge and compares displaced solves with opposing-face Coulomb integrals, refined from 48 to 80 Gauss points per axis. It covers paired and triple half-cell shifts, both floating-point types, two grids and Lorentz factors, full/pruned FFTs, caching, and exported-solver controls. The Python launcher uses only the standard library.
    """

    from _scibmad import run_julia

    _PROGRAM = r'''
    using SciBmad, SpaceCharge, LinearAlgebra, Printf
    const SC = SpaceCharge
    const BAD = Ref(false)
    const H = 2.0^-10

    function face_field(offset, gamma, order)
        # Integrate -d(1/r)/d r_i in direction i first. The remaining two smooth
        # opposing-face integrals avoid both the indefinite primitive and the FFT.
        # Golub-Welsch supplies Gauss-Legendre nodes/weights from a symmetric matrix.
        rule = eigen(SymTridiagonal(zeros(order), [k/sqrt(4k*k-1) for k in 1:order-1]))
        nodes, weights = rule.values, 2 .* rule.vectors[1,:].^2
        widths = [H,H,gamma*H]
        center = widths .* ([1.,0.,0.] .+ collect(offset))
        lo, hi = center-widths/2, center+widths/2
        result = zeros(3)
        for axis in 1:3
            j,k = [a for a in 1:3 if a != axis]
            integral = 0.
            for a in 1:order, b in 1:order
                y = (lo[j]+hi[j])/2 + widths[j]/2*nodes[a]
                z = (lo[k]+hi[k])/2 + widths[k]/2*nodes[b]
                square = y*y+z*z
                integral += weights[a]*weights[b]*(1/sqrt(lo[axis]^2+square)-1/sqrt(hi[axis]^2+square))
            end
            integral *= widths[j]*widths[k]/4
            # rho stores node charge. Longitudinal widths stretch by gamma;
            # transverse fields acquire gamma, exactly as the solver contract says.
            result[axis] = SC.FPEI*1e-12/prod(widths)*integral*(axis < 3 ? gamma : 1.)
        end
        result
    end

    function mesh_case(T,n,gamma)
        h = T(H)
        mesh = Mesh3D(n,(zero(T),zero(T),zero(T)),Tuple(T.(n.-1).*h);T,gamma=T(gamma))
        deposit!(mesh,T[0],T[0],T[0],T[1e-12])
        @assert mesh.delta == (h,h,h) && sum(mesh.rho) == T(1e-12)
        mesh
    end

    function main()
        BLAS.set_num_threads(1)
        SC.FFTW.set_num_threads(1)
        @assert Threads.nthreads() == BLAS.get_num_threads() == SC.FFTW.get_num_threads() == 1
        println("ENV Julia=",VERSION," SpaceCharge=",pkgversion(SC)," SciBmad=",pkgversion(SciBmad)," numerical_threads=1")
        offsets = ((0.,0.,0.),(.5,0.,0.),(0.,.5,0.),(0.,0.,.5),(1.,1.,0.),
                   (.5,.5,0.),(.5,0.,.5),(0.,.5,.5),(.5,.5,.5),(.25,.25,.25))
        combinations, failed, max_refinement = 0,0,0.
        for gamma in (1.,3.), offset in offsets
            reference = face_field(offset,gamma,80)
            refinement = maximum(abs.(reference-face_field(offset,gamma,48)))
            @assert refinement < 2e-10
            max_refinement = max(max_refinement,refinement)
            half_pair = count(==(.5),offset) >= 2
            group_failed, sample, nf = 0,zeros(3),0
            for T in (Float64,Float32), n in ((4,4,4),(5,4,3)), cache in (false,true), pruned in (false,true)
                mesh = mesh_case(T,n,gamma)
                SC.solve_freespace!(mesh;offset=Tuple(T.(offset).*T(H)),cache_green=cache,
                                   pruned_fft=pruned,fft_flags=SC.FFTW.ESTIMATE)
                nonfinite = count(!isfinite,mesh.efield)
                observed = Float64.(vec(mesh.efield[2,1,1,:]))
                budget = 512*eps(T)*maximum(abs,reference)
                discrepancy = nonfinite != 0 || maximum(abs.(observed-reference)) > budget
                !half_pair && discrepancy && error("control failed: $offset $gamma $T $n")
                group_failed += discrepancy
                failed += discrepancy
                combinations += 1
                if T == Float64 && n == (4,4,4) && !cache && !pruned
                    sample, nf = observed, nonfinite
                end
            end
            BAD[] |= group_failed > 0
            @printf("OFFSET gamma=%g offset_cells=%s variants=16 failed=%d sample_nonfinite=%d/192 expected_nonfinite=0 observed_V_per_m=%s expected_V_per_m=%s refinement=%.3g derivation=opposing_face_Coulomb_integrals\n",
                    gamma,string(offset),group_failed,nf,string(sample),string(reference),refinement)
        end
        # Exported solve! only supplies zero transverse offsets. Its cathode image
        # displacement is exactly 3.5 longitudinal cells, exercising a single zero.
        for cathode in (false,true), T in (Float64,Float32)
            mesh = mesh_case(T,(4,4,4),1)
            h = T(H)
            mesh.min_bounds = (zero(T),zero(T),T(.25)*h)
            mesh.max_bounds = (T(3)*h,T(3)*h,T(3.25)*h)
            solve!(mesh;at_cathode=cathode,fft_flags=SC.FFTW.ESTIMATE)
            @assert all(isfinite,mesh.efield)
            println("EXPORTED_CONTROL T=",T," cathode=",cathode," nonfinite=0 expected=0 derivation=zero_transverse_offsets")
        end
        @printf("TOTAL combinations=%d failed=%d expected_failed=0 max_oracle_refinement_V_per_m=%.3g\n",combinations,failed,max_refinement)
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
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | SpaceCharge 1.3.0 (`ae30463305729aec2f04d4687ffd547b94750cb2`), source (SciBmad 0.6.0 5f0fb83ac889), Julia 1.13.2, CPU, Float64, one thread |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | SpaceCharge 1.3.0 (`ae30463305729aec2f04d4687ffd547b94750cb2`), source (SciBmad 0.6.0 5f0fb83ac889), Julia 1.13.2, CPU, Float32/Float64, one thread |

