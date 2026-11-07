---
title: "OPT-03: h<ijkl> columns are degree-weighted, and the scalar and Taylor paths disagree"
hide:
  - toc
---

# OPT-03 — `h<ijkl>` columns are degree-weighted, and the scalar and Taylor paths disagree

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P1 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on SciBmad release [v0.6.0](https://github.com/bmad-sim/SciBmad.jl/tree/v0.6.0) (`62399f9`) |
| Issue | [[gh:https://github.com/bmad-sim/SciBmad.jl/issues/156]], filed from this evidence |
| Pull request | [[gh:https://github.com/bmad-sim/SciBmad.jl/pull/157]], filed from this evidence |
| Source | [`src/twiss/twiss_operators.jl:1066-1085`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L1066-L1085)<br>[`twiss_operators.jl:1026-1063`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L1026-L1063)<br>[`twiss_operators.jl:1052`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/twiss/twiss_operators.jl#L1052)<br>[`NonlinearNormalForm.jl/src/bengtsson.jl:37-55`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/bengtsson.jl#L37-L55)<br>[`docs/src/quickstart.md:268-282`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/quickstart.md#L268-L282) |

The `twiss` resonance-driving-term columns `h<ijkl>`, the Bengtsson coefficients of the one-turn Lie generator h, are scaled by a monomial-dependent factor, so scalar and Taylor-series outputs disagree. For a thin octupole of integrated strength K3L the documented coefficient is `h2200` = K3L β²/16, and the normal-form tune slope with amplitude independently confirms this normalization. With K3L = 1000 m⁻³ in a thin-quadrupole cell, the octupole's `h2200` increment is 2961 m⁻¹ in scalar output and 1480 m⁻¹ in Taylor output instead of 740.1 m⁻¹, and a thin sextupole gives `h3000` three times its expected value. Scalar coefficients carry the monomial degree as a factor, Taylor coefficients the last contributing exponent, and the chromatic `dh…_n` columns inherit the error.

Run `julia OPT-03.jl`.

??? example "Reproducer: OPT-03.jl"

    [Download OPT-03.jl](OPT-03.jl)

    ```julia
    # OPT-03: `h<ijkl>` columns are degree-weighted, and the scalar and Taylor paths disagree
    #
    # The `twiss` resonance-driving-term columns `h<ijkl>`, the Bengtsson coefficients of the one-turn
    # Lie generator h, are scaled by a monomial-dependent factor, so scalar and Taylor-series outputs
    # disagree. For a thin octupole of integrated strength K3L the documented coefficient is `h2200` =
    # K3L β²/16, and the normal-form tune slope with amplitude independently confirms this
    # normalization. With K3L = 1000 m⁻³ in a thin-quadrupole cell, the octupole's `h2200` increment is
    # 2961 m⁻¹ in scalar output and 1480 m⁻¹ in Taylor output instead of 740.1 m⁻¹, and a thin sextupole
    # gives `h3000` three times its expected value. Scalar coefficients carry the monomial degree as a
    # factor, Taylor coefficients the last contributing exponent, and the chromatic `dh…_n` columns
    # inherit the error.
    #
    # The script places a thin octupole (K3L = 1000 m⁻³) in front of a periodic thin-quadrupole cell and
    # compares the octupole's increment of `h2200` over the bare cell, in scalar and Taylor output, with
    # K3L β²/16; subtracting the bare cell removes the drifts' own fourth-order terms. The normal-form
    # detuning dQ₁/dJ₁, which equals c/π for H = cJ², checks that value independently. It then reads
    # `h3000` and `h2100` for a thin sextupole (K2L = 10 m⁻²) and compares their magnitudes with K2L
    # β^(3/2)/(12√2) and three times that.
    using SciBmad
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    mk(e...) = Beamline([e..., qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)
    K3 = 1000.0; K2 = 10.0
    base = mk(); oct = mk(Multipole(Kn3L = K3)); sext = mk(Multipole(Kn2L = K2))

    t0 = twiss(base, rf_on = false, order = 3, at = [1], cols = ["h2200", "beta1"])
    β = t0.beta1[1]
    # Phasors a± = (Q ± iP)/√2, J = a₊a₋ and x = √β (a₊ + a₋)/√2, so V = K3L x⁴/24 contributes c·J² with
    # c = K3L β²/16; H = cJ² detunes by dQ/dJ = c/π.
    c = K3 * β^2 / 16
    bad = false
    # A degree-d monomial c·u^m contributes ±i c m_k u^(m−e_k) to vector-field component k; the scalar
    # output sums the components (d·c), the Taylor output keeps only the last one (m_k·c).
    for tps in (false, true)
        a = twiss(base, rf_on = false, order = 3, at = [1], cols = ["h2200"], as_taylor_series = tps)
        b = twiss(oct, rf_on = false, order = 3, at = [1], cols = ["h2200"], as_taylor_series = tps)
        dh = scalar(b.h2200[1]) - scalar(a.h2200[1])
        println("as_taylor_series=$tps  h2200 increment=", dh, "  expected=", c, "  ratio=", real(dh) / c)
        global bad |= abs(abs(dh) - c) > 1e-9 * c
    end
    ta = twiss(base, rf_on = false, order = 3, at = [1], cols = ["beta1"])
    tb = twiss(oct, rf_on = false, order = 3, at = [1], cols = ["beta1"])
    dQdJ = getterm(tb.q1, J1 = 1) - getterm(ta.q1, J1 = 1)
    println("independent: normal-form dQ1/dJ1 increment=", dQdJ, "  π·dQ/dJ=", pi * dQdJ, "  K3Lβ²/(16π)=", K3 * β^2 / (16pi))

    # V = K2L x³/6 gives |h3000| = K2L β^(3/2)/(6·2√2) and |h2100| = 3|h3000|.
    h30x = K2 * β^1.5 / (6 * 2 * sqrt(2))
    for tps in (false, true)
        s = twiss(sext, rf_on = false, order = 2, at = [1], cols = ["h3000", "h2100"], as_taylor_series = tps)
        h30 = abs(scalar(s.h3000[1])); h21 = abs(scalar(s.h2100[1]))
        println("as_taylor_series=$tps  |h3000|=", h30, " (ratio ", h30 / h30x, ")  |h2100|=", h21, " (ratio ", h21 / (3 * h30x), ")")
        global bad |= abs(h30 - h30x) > 1e-9 * h30x
    end
    println("expected |h3000|=", h30x, "  |h2100|=", 3 * h30x)
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |

