---
title: "OPT-34: Full-circle dynamic-aperture scans abort during analysis"
hide:
  - toc
---

# OPT-34 — Full-circle dynamic-aperture scans abort during analysis

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on SciBmad main (`ae9423d`, head build) |
| Source | [`src/dynamic_aperture.jl:189`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/dynamic_aperture.jl#L189)<br>[`:193`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/dynamic_aperture.jl#L193)<br>[`:32-33`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/dynamic_aperture.jl#L32-L33) |

`dynamic_aperture` with a polar scan range outside [−π, π], such as `theta_lims=(0, 2π)` or (2π, 3π), aborts after tracking with `Something went wrong with the analysis`. The cause is a sanity check that compares the principal angle `atan(y, x)` with the requested angle as given. Directions are periodic modulo 2π, the docstring does not restrict the range, and the same scan over (−π, π) returns infinite acceptance on an aperture-free orthogonal one-turn map. At the requested angle 3π/2 (4.712 rad), the principal angle is −1.571 rad, so the check fails after all four turns have been tracked. Scans within [−π, π] are unaffected.

Run `julia OPT-34.jl`.

??? example "Reproducer: OPT-34.jl"

    [Download OPT-34.jl](OPT-34.jl)

    ```julia
    # OPT-34: Full-circle dynamic-aperture scans abort during analysis
    #
    # `dynamic_aperture` with a polar scan range outside [−π, π], such as `theta_lims=(0, 2π)` or (2π,
    # 3π), aborts after tracking with `Something went wrong with the analysis`. The cause is a sanity
    # check that compares the principal angle `atan(y, x)` with the requested angle as given. Directions
    # are periodic modulo 2π, the docstring does not restrict the range, and the same scan over (−π, π)
    # returns infinite acceptance on an aperture-free orthogonal one-turn map. At the requested angle
    # 3π/2 (4.712 rad), the principal angle is −1.571 rad, so the check fails after all four turns have
    # been tracked. Scans within [−π, π] are unaffected.
    #
    # The script wraps an uncoupled six-dimensional rotation map with angles (0.29, 0.51, 0.83) in a
    # user transport element with no apertures and runs `dynamic_aperture` with four radial and nine
    # angular samples and four turns over `theta_lims` (0, π), (−π, π), (0, 2π) and (2π, 3π). The two
    # principal ranges must return infinite acceptance, the other two are expected to throw the analysis
    # error, and the script prints the principal `atan` of the direction 3π/2. It runs in a temporary
    # directory, which holds and then removes the diagnostic files that the failing call writes.
    using SciBmad, LinearAlgebra, Printf
    BLAS.set_num_threads(1)
    rotation(t) = [cos(t) sin(t); -sin(t) cos(t)]
    function linear_transport(v,q,matrix)
        (ntuple(i -> sum(matrix[i,j]*v[j] for j in 1:6),6),q)
    end
    function main()
        M=zeros(6,6)
        for (i,theta) in enumerate((.29,.51,.83))
            M[2i-1:2i,2i-1:2i] = rotation(theta)
        end
        @assert norm(M'*M-I) < 1e-15
        ring=Beamline([LineElement(L=1.,transport_map=linear_transport,transport_map_params=M)],species_ref=Species("electron"),pc_ref=1e9)
        for (label,limits,expected_failure) in (("upper_half",(0.,pi),false),("principal_full",(-pi,pi),false),("unwrapped_full",(0.,2pi),true),("shifted_half",(2pi,3pi),true))
            println("SCAN ",label," theta_lims=",limits)
            failed=false
            try
                xx,yy = dynamic_aperture(ring;n_r=4,n_theta=9,deltas=[0.],max_sig_x=3.,max_sig_y=2.,emit_1=2e-6,emit_2=3e-6,n_turns=4,theta_lims=limits,output_file=label*".csv")
                println("RETURN all_infinite=",all(isinf,xx)&&all(isinf,yy))
                @assert all(isinf,xx)&&all(isinf,yy)
            catch e
                e isa ErrorException || rethrow()
                occursin("Something went wrong with the analysis",e.msg) || rethrow()
                failed=true
                println("EXPECTED_ANALYSIS_EXCEPTION ",e.msg)
            end
            @assert failed == expected_failure
        end
        @printf("3pi/2 principal_atan=%.16g requested=%.16g difference=%.16g\n",atan(sin(3pi/2),cos(3pi/2)),3pi/2,3pi/2-atan(sin(3pi/2),cos(3pi/2)))
        println("DEFECT REPRODUCED")
    end
    # The failing API writes error.dlm; contain and clean up its diagnostic files.
    mktempdir(; prefix="OPT-34-") do output_dir
        cd(output_dir) do
            main()
        end
    end

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | SciBmad main (`5f0fb83ac889`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | SciBmad main (`5f0fb83ac889`), source, Julia 1.13.2, CPU, Float64, serial |

