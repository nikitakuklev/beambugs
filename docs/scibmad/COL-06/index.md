---
title: "COL-06: track_spin does not validate a matrix initial spin"
hide:
  - toc
---

# COL-06 — `track_spin` does not validate a matrix initial spin

| | |
|---|---|
| Code | [SciBmad](../index.md) · Collective effects and spin |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad commit [`ae9423d`](https://github.com/bmad-sim/SciBmad.jl/commit/ae9423d703b4), committed 2026-10-28 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`SciBmad.jl/src/track.jl:322-350`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L322-L350)<br>[`track.jl:305-307`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L305-L307)<br>[`track.jl:327-328`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L327-L328)<br>[`track.jl:340-350`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L340-L350) |

The postprocessing helper `track_spin`, which applies tracked spin quaternions to initial spins, accepts an initial-spin matrix of the wrong shape and silently returns wrong spins that are not unit vectors. Its documented input is one three-component row per particle, and identity quaternions must return those spins unchanged; column-major indexing of the misshaped matrix predicts exactly the wrong result. For four particles, a transposed 3×4 matrix is accepted and turns the fourth spin from the intended (0.6000, 0.8000, 0) into (0, 0, 0.6000), shortening it from length 1.000 to 0.6000 instead of rejecting the shape. Correctly shaped input works; source inspection also finds unchecked out-of-bounds reads for undersized matrices.

Run `julia COL-06.jl`.

??? example "Reproducer: COL-06.jl"

    [Download COL-06.jl](COL-06.jl)

    ```julia
    # COL-06: `track_spin` does not validate a matrix initial spin
    #
    # The postprocessing helper `track_spin`, which applies tracked spin quaternions to initial spins,
    # accepts an initial-spin matrix of the wrong shape and silently returns wrong spins that are not
    # unit vectors. Its documented input is one three-component row per particle, and identity
    # quaternions must return those spins unchanged; column-major indexing of the misshaped matrix
    # predicts exactly the wrong result. For four particles, a transposed 3×4 matrix is accepted and
    # turns the fourth spin from the intended (0.6000, 0.8000, 0) into (0, 0, 0.6000), shortening it
    # from length 1.000 to 0.6000 instead of rejecting the shape. Correctly shaped input works; source
    # inspection also finds unchecked out-of-bounds reads for undersized matrices.
    #
    # The script passes four initial spins with identity quaternions to `track_spin`, once correctly
    # shaped (4×3, one row per particle) and once transposed (3×4), and prints both results and the
    # lengths of the returned spins. As an independent check, it predicts the fourth particle's spin
    # from Julia's column-major indexing of the transposed matrix. Every read stays inside the
    # 12-element matrix, so this misuse is memory-safe.
    using SciBmad

    n = 4
    q = zeros(n, 4, 1); q[:, 1, 1] .= 1.0                 # identity rotation for every particle
    S = [1.0 0 0; 0 1.0 0; 0 0 1.0; 0.6 0.8 0]            # intended initial spins, one row each
    s_ok = track_spin(q, S)
    # A matrix s0 must have size (n_particles, 3). The matrix branch of track_spin! re-checks
    # size(s,1) == size(q,1) instead of the shape of s0, and the kernel reads s0[i,1:3] under
    # @inbounds. Expected: an AssertionError, or output identical to the correctly shaped input.
    local s_bad
    accepted = try
        global s_bad = track_spin(q, permutedims(S))       # (3, 4): wrong shape
        true
    catch err
        println("rejected: ", typeof(err)); false
    end
    println("correct input   s[:, :, 1] = ", [s_ok[i, :, 1] for i in 1:n])
    if accepted
        println("transposed input accepted, s[:, :, 1] = ", [s_bad[i, :, 1] for i in 1:n])
        println("unit-length check of returned spins: ", [round(sqrt(sum(abs2, s_bad[i, :, 1])), digits = 6) for i in 1:n])
    end
    # Independent recheck: column-major prediction s[i, j] = vec(M)[i + 3(j-1)] for M = permutedims(S).
    lin = vec(permutedims(S))
    println("recheck: predicted particle-4 spin from linear indexing = ", [lin[4 + 3 * (j - 1)] for j in 1:3])
    println(accepted && s_bad != s_ok ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), source, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), source, Julia 1.13.2, CPU, Float64, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), source, Julia 1.13.2, CPU, Float64, serial |

