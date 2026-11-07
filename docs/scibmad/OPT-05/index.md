---
title: "OPT-05: Map-valued E1..E3 throw UndefVarError: nj_mat"
hide:
  - toc
---

# OPT-05 — Map-valued `E1..E3` throw `UndefVarError: nj_mat`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad commit [`ae9423d`](https://github.com/bmad-sim/SciBmad.jl/commit/ae9423d703b4), committed 2026-10-28 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`src/twiss/twiss_operators.jl:210-223`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss_operators.jl#L210-L223)<br>[`haskey(cache.persistent_matrix, nj_mat)`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss_operators.jl#L217)<br>[`twiss_operators.jl:220`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss_operators.jl#L220)<br>[`twiss_operators.jl:218`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss_operators.jl#L218)<br>[`twiss.jl:188-196`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L188-L196)<br>[`twiss_operators.jl:122`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss_operators.jl#L122) |

Requesting the De Moivre matrix `E1` from coasting `twiss` above order 1, such as `twiss(chrom=2)`, throws `UndefVarError: nj_mat` instead of returning the documented matrix, because its map-valued branch reads undefined variables. For a thin-quadrupole test cell, both −B₁S from the working `B1` output (S the symplectic form) and the Twiss form [β −α; −α γ] give E₁ = [3.441 1.376; 1.376 0.8412], with β in m and γ in m⁻¹. By source inspection, `E1` and `E2` fail in higher-order coasting runs, and `E1`–`E3` in higher-order RF runs with knob parameters. Order-1 optics and RF optics without knobs take a separate linear branch; `E3` is explicitly unavailable for coasting beams.

Run `julia OPT-05.jl`.

??? example "Reproducer: OPT-05.jl"

    [Download OPT-05.jl](OPT-05.jl)

    ```julia
    # OPT-05: Map-valued `E1..E3` throw `UndefVarError: nj_mat`
    #
    # Requesting the De Moivre matrix `E1` from coasting `twiss` above order 1, such as
    # `twiss(chrom=2)`, throws `UndefVarError: nj_mat` instead of returning the documented matrix,
    # because its map-valued branch reads undefined variables. For a thin-quadrupole test cell, both
    # −B₁S from the working `B1` output (S the symplectic form) and the Twiss form [β −α; −α γ] give E₁
    # = [3.441 1.376; 1.376 0.8412], with β in m and γ in m⁻¹. By source inspection, `E1` and `E2` fail
    # in higher-order coasting runs, and `E1`–`E3` in higher-order RF runs with knob parameters. Order-1
    # optics and RF optics without knobs take a separate linear branch; `E3` is explicitly unavailable
    # for coasting beams.
    #
    # The script runs `twiss(chrom=2)` without RF on a periodic cell of thin quadrupoles and 1 m drifts
    # (1 GeV/c electrons), reads `B1` from the map-valued branch together with `beta1` and `alpha1`, and
    # builds the expected E₁ both as −B₁S and from the Twiss form [β −α; −α γ]. It then requests `E1` in
    # the same configuration and reports the error it throws.
    using SciBmad, LinearAlgebra
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    bl = Beamline([qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)

    tb = twiss(bl, rf_on = false, chrom = 2, at = [1], cols = ["beta1", "alpha1", "B1"])
    # Expected E₁ = B₁ ∘ (−S), at δ = 0 the matrix −B₁S, with B₁ from the map-valued B branch.
    B = tb.B1[1]; S = kron(I(2), [0.0 1; -1 0])
    β, α = tb.beta1[1], tb.alpha1[1]
    println("chrom=2 B1 (map branch) [1:2,1:2] = ", B[1:2, 1:2], "  -> expected E1 = -B1*S: ", (-B*S)[1:2, 1:2])
    println("independent Twiss form [β -α; -α γ] = ", [β -α; -α (1 + α^2)/β])
    msg = try
        twiss(bl, rf_on = false, chrom = 2, at = [1], cols = ["E1"])
        "no error"
    catch err
        sprint(showerror, err)
    end
    println("twiss(chrom=2, cols=[\"E1\"]) -> ", first(split(msg, '\n')))
    println(occursin("nj_mat", msg) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |

