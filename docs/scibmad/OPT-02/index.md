---
title: "OPT-02: Scalar d<col>_<n> returns f⁽ⁿ⁾/n, not the nth derivative"
hide:
  - toc
---

# OPT-02 — Scalar `d<col>_<n>` returns f⁽ⁿ⁾/n, not the nth derivative

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P1 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on SciBmad release [v0.6.0](https://github.com/bmad-sim/SciBmad.jl/tree/v0.6.0) (`62399f9`) |
| Issue | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/154]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/155]], filed from this evidence |
| Source | [`src/twiss/twiss_map.jl:134-146`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_map.jl#L134-L146)<br>[`twiss.jl:75-82`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L75-L82)<br>[`136-138`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss.jl#L136-L138) |

A `twiss` chromatic derivative column `d<col>_<n>` with scalar output, such as `dx_2` or `dbeta1_2`, returns the nth derivative with respect to the momentum offset δ divided by n. The docstring promises the nth derivative itself, n! times the stored Taylor coefficient, which the `as_taylor_series=true` output, nested ForwardDiff and finite differences of tracking all give. For a thin-quadrupole cell with `chrom=4`, `dbeta1_2` is 2.585 m instead of 5.171 m, and `dbeta1_3` is −7.457 m instead of −22.37 m. Every scalar column of order 2 and higher is low by exactly n; first-order columns such as `dx` and `dbeta1` are correct.

Run `julia OPT-02.jl`.

??? example "Reproducer: OPT-02.jl"

    [Download OPT-02.jl](OPT-02.jl)

    ```julia
    # OPT-02: Scalar `d<col>_<n>` returns f⁽ⁿ⁾/n, not the nth derivative
    #
    # A `twiss` chromatic derivative column `d<col>_<n>` with scalar output, such as `dx_2` or
    # `dbeta1_2`, returns the nth derivative with respect to the momentum offset δ divided by n. The
    # docstring promises the nth derivative itself, n! times the stored Taylor coefficient, which the
    # `as_taylor_series=true` output, nested ForwardDiff and finite differences of tracking all give.
    # For a thin-quadrupole cell with `chrom=4`, `dbeta1_2` is 2.585 m instead of 5.171 m, and
    # `dbeta1_3` is −7.457 m instead of −22.37 m. Every scalar column of order 2 and higher is low by
    # exactly n; first-order columns such as `dx` and `dbeta1` are correct.
    #
    # The script runs `twiss(chrom=4)` on a periodic cell of thin quadrupoles and 1 m drifts (1 GeV/c
    # electrons, RF off) and reads `dbeta1_2` and `dbeta1_3` with scalar and Taylor-series output. It
    # compares them with nested ForwardDiff derivatives of the analytic thin-lens β₁(δ), and checks
    # those against five-point finite differences in δ of β₁ from tracked one-turn Jacobians.
    using SciBmad, LinearAlgebra
    const FD = SciBmad.ForwardDiff

    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    bl = Beamline([qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)
    # Analytic thin-lens β₁(δ), drifts with the exact-drift chromatic length L/(1+δ); the nth
    # δ-derivative of a Taylor series is n! times its nth coefficient.
    D(δ) = [1 1 / (1 + δ); 0 one(δ)]
    K(k) = [1.0 0; -k 1]
    beta(m) = m[1, 2] / sin(acos(tr(m) / 2))
    βa(δ) = beta(D(δ) * K(-0.7) * D(δ) * K(0.8))
    d2 = FD.derivative(x -> FD.derivative(βa, x), 0.0)
    d3 = FD.derivative(y -> FD.derivative(x -> FD.derivative(βa, x), y), 0.0)

    cols = ["dbeta1_2", "dbeta1_3"]
    ts = twiss(bl, rf_on = false, chrom = 4, cols = cols)
    tt = twiss(bl, rf_on = false, chrom = 4, cols = cols, as_taylor_series = true)
    println("dbeta1_2: scalar=", ts.dbeta1_2[1], "  taylor scalar part=", scalar(tt.dbeta1_2[1]), "  analytic=", d2,
            "  analytic/scalar=", d2 / ts.dbeta1_2[1])
    println("dbeta1_3: scalar=", ts.dbeta1_3[1], "  taylor scalar part=", scalar(tt.dbeta1_3[1]), "  analytic=", d3,
            "  analytic/scalar=", d3 / ts.dbeta1_3[1])

    # Independent check: beta from tracked one-turn Jacobians, 5-point stencils in δ.
    βt(δ) = beta(FD.jacobian(v -> vec(track(bl, v0 = reshape(v, 1, 6)).v[1, :, end]), [0, 0, 0, 0, 0, δ])[1:2, 1:2])
    h = 2e-3
    b = [βt(k * h) for k in -2:2]
    fd2 = (-b[1] + 16b[2] - 30b[3] + 16b[4] - b[5]) / (12h^2)
    fd3 = (-b[1] + 2b[2] - 2b[4] + b[5]) / (2h^3)
    println("tracking finite differences (h=$h): d2=", fd2, " d3=", fd3)
    bad = abs(ts.dbeta1_2[1] - d2) > 1e-6 * abs(d2) || abs(ts.dbeta1_3[1] - d3) > 1e-6 * abs(d3)
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |

