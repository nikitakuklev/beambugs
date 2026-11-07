---
title: "OPT-08: Row 2 of the first table column reuses row 1's cache"
hide:
  - toc
---

# OPT-08 — Row 2 of the first table column reuses row 1's cache

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`, head build) |
| Source | [`src/twiss/twiss_df.jl:24-26`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss_df.jl#L24-L26)<br>[`twiss_df.jl:47-58`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss_df.jl#L47-L58)<br>[`twiss.jl:118-120`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L118-L120) |

When the first column of a `twiss` table reads the per-row optics cache, for example with `base_cols=[]` or a custom `base_cols` starting with an optics column, row 2 of that column repeats row 1. Each row must describe its own sample, and drift transport of the start values, β(s) = β₀ − 2α₀s + γ₀s², gives the expected value. In a lattice of thin quadrupoles and 1 m drifts, `beta1` at s = 1 m reads 1.529 m, the value at s = 0, instead of 3.441 m, while the `alpha1` column of the same call is correct. The default `base_cols` start with `"index"`, which does not use the cache, so default tables are unaffected.

Run `julia OPT-08.jl`.

??? example "Reproducer: OPT-08.jl"

    [Download OPT-08.jl](OPT-08.jl)

    ```julia
    # OPT-08: Row 2 of the first table column reuses row 1's cache
    #
    # When the first column of a `twiss` table reads the per-row optics cache, for example with
    # `base_cols=[]` or a custom `base_cols` starting with an optics column, row 2 of that column
    # repeats row 1. Each row must describe its own sample, and drift transport of the start values,
    # β(s) = β₀ − 2α₀s + γ₀s², gives the expected value. In a lattice of thin quadrupoles and 1 m
    # drifts, `beta1` at s = 1 m reads 1.529 m, the value at s = 0, instead of 3.441 m, while the
    # `alpha1` column of the same call is correct. The default `base_cols` start with `"index"`, which
    # does not use the cache, so default tables are unaffected.
    #
    # The script runs `twiss` on the lattice drift, thin quadrupole, drift, thin quadrupole (1 GeV/c
    # electrons, RF off) with the columns `beta1` and `alpha1`, once with the default `base_cols` and
    # once with `base_cols=[]`, and compares the two `beta1` columns row by row. It checks row 2 against
    # drift transport of the row-1 Twiss parameters to s = 1 m.
    using SciBmad
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    bl = Beamline([d, qf, d, qd], species_ref = Species("electron"), pc_ref = 1e9)

    ref = twiss(bl, rf_on = false, cols = ["beta1", "alpha1"])                 # base cols first
    # With base_cols=[] the first column is beta1, which reads the per-row cache; the default base
    # columns start with "index", which does not.
    got = twiss(bl, rf_on = false, base_cols = String[], cols = ["beta1", "alpha1"])
    println("rows s=", ref.s[1:3])
    println("beta1 with base_cols   : ", ref.beta1[1:3])
    println("beta1 with base_cols=[]: ", got.beta1[1:3])
    println("alpha1 with base_cols   : ", ref.alpha1[1:3])
    println("alpha1 with base_cols=[]: ", got.alpha1[1:3])
    β0, α0 = ref.beta1[1], ref.alpha1[1]; γ0 = (1 + α0^2) / β0; s2 = ref.s[2]
    println("independent: drift transport beta1 at s=", s2, " is ", β0 - 2α0 * s2 + γ0 * s2^2)
    diff = maximum(abs, got.beta1 .- ref.beta1)
    println("max |beta1 difference| = ", diff)
    println(diff > 1e-12 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |

