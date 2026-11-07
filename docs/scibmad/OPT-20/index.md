---
title: "OPT-20: getterm returns 0.0 for summary coefficients the normal form never computes"
hide:
  - toc
---

# OPT-20 — `getterm` returns 0.0 for summary coefficients the normal form never computes

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P1 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-23 on SciBmad commit [`5f0fb83`](https://github.com/bmad-sim/SciBmad.jl/commit/5f0fb83ac889), committed 2026-10-15 |
| Source | [`src/adv.jl:28-30`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/adv.jl#L28-L30)<br>[`NonlinearNormalForm.jl/src/normal.jl:699`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L699)<br>[`707`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L707)<br>[`:814`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L814)<br>[`GTPSA.jl/src/methods.jl:316-321`](https://github.com/bmad-sim/GTPSA.jl/blob/a5790a7deaa62c13d3edc168d5f98b15e8a3c490/src/methods.jl#L316-L321)<br>[`SciBmad.jl/src/twiss/twiss.jl:750-756`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L750-L756)<br>[`adv.jl:29`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/adv.jl#L29)<br>[`twiss.jl:102`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L102)<br>[`docs/src/quickstart.md:264-266`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/quickstart.md#L264-L266) |

`getterm` returns 0.0 for coefficients of the summary tunes, `etac` and `alphac` that the descriptor admits but the normal form never computes, because its guard checks only the requested monomial. A tune coefficient needs a map term one degree higher, and a coasting `etac` or `alphac` coefficient needs one more power of the momentum deviation δ. For a 1 GeV/c proton thin-quadrupole cell, `twiss(chrom=2)` reports the δ² coefficient of `q1` as 0.0 instead of 0.1462, confirmed by analytic differentiation and tracking. A thin octupole with K3L = 1000 m⁻³ adds zero detuning dQ₁/dJ₁ (J₁ the mode-1 action) at `order=2` instead of 235.6 m⁻¹, the perturbative increment recovered at order 3. Both insufficient total order and restrictive per-variable caps cause these zeros.

Run `julia OPT-20.jl`.

??? example "Reproducer: OPT-20.jl"

    [Download OPT-20.jl](OPT-20.jl)

    ```julia
    # OPT-20: `getterm` returns 0.0 for summary coefficients the normal form never computes
    #
    # `getterm` returns 0.0 for coefficients of the summary tunes, `etac` and `alphac` that the
    # descriptor admits but the normal form never computes, because its guard checks only the requested
    # monomial. A tune coefficient needs a map term one degree higher, and a coasting `etac` or `alphac`
    # coefficient needs one more power of the momentum deviation δ. For a 1 GeV/c proton thin-quadrupole
    # cell, `twiss(chrom=2)` reports the δ² coefficient of `q1` as 0.0 instead of 0.1462, confirmed by
    # analytic differentiation and tracking. A thin octupole with K3L = 1000 m⁻³ adds zero detuning
    # dQ₁/dJ₁ (J₁ the mode-1 action) at `order=2` instead of 235.6 m⁻¹, the perturbative increment
    # recovered at order 3. Both insufficient total order and restrictive per-variable caps cause these
    # zeros.
    #
    # The script runs `twiss` on the coasting cell [Multipole(Kn1L=0.8), Drift(1), Multipole(Kn1L=−0.7),
    # Drift(1)] at `chrom=2` and 3 and compares the δ² and δ³ coefficients of `q1` and the δ²
    # coefficients of `etac` and `alphac` with nested ForwardDiff derivatives of the analytic tune,
    # five-point differences of tracked tunes and derivatives of the tracked one-turn z. It compares the
    # octupole detuning increment with K3L βₓ²/(16π) at orders 2 and 3, prints the bunched (RF on)
    # action coefficients at both orders, and checks for each request whether the requested monomial and
    # the map term it is read from are valid in the descriptor.
    using SciBmad, LinearAlgebra
    const FD = SciBmad.ForwardDiff
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
    cell = Beamline([qf, d, qd, d], species_ref = Species("proton"), pc_ref = 1e9)
    # The closed orbit is zero at every δ and the thin kicks in px do not depend on δ, so the one-turn
    # matrix uses the drift matrix [1 L/(1+δ); 0 1] and q1(δ) = acos(tr M(δ)/2)/2π; nested ForwardDiff
    # derivatives of it give the expected coefficients.
    M(δ) = [1 1 / (1 + δ); 0 one(δ)] * [1.0 0; 0.7 1] * [1 1 / (1 + δ); 0 one(δ)] * [1.0 0; -0.8 1]
    qa(δ) = acos(tr(M(δ)) / 2) / 2pi
    c2 = FD.derivative(x -> FD.derivative(qa, x), 0.0) / 2

    t2 = twiss(cell, rf_on = false, chrom = 2, at = [])
    t3 = twiss(cell, rf_on = false, chrom = 3, at = [])
    println("descriptor of twiss(chrom=2): ", GTPSA.getdesc(t2.q1.taylor_series))
    println("q1 δ² : chrom=2 ", getterm(t2.q1, delta = 2), "  chrom=3 ", getterm(t3.q1, delta = 2), "  analytic ", c2)
    println("q1 δ³ : chrom=3 ", getterm(t3.q1, delta = 3), "  analytic ",
            FD.derivative(y -> FD.derivative(x -> FD.derivative(qa, x), y), 0.0) / 6)
    q1t(δ) = acos(tr(FD.jacobian(u -> vec(track(cell, v0 = reshape(u, 1, 6), rf_on = false).v[1, :, end]),
                                 [0, 0, 0, 0, 0, δ])[1:2, 1:2]) / 2) / 2pi
    h = 1e-3; f = [q1t(k * h) for k in -2:2]
    println("independent: tracked q1 five-point δ² coefficient (h=$h) = ", (-f[1] + 16f[2] - 30f[3] + 16f[4] - f[5]) / (24h^2))

    # Coordinate slip etac(δ) = −(1/C) dz/dδ with z the tracked one-turn z and C = 2 m: its δ²
    # coefficient is −3 z₃/C, with z₃ the δ³ coefficient of z.
    z1(δ) = track(cell, v0 = reshape([0, 0, 0, 0, 0, δ], 1, 6), rf_on = false).v[1, 5, end]
    z3 = FD.derivative(z -> FD.derivative(y -> FD.derivative(z1, y), z), 0.0) / 6
    println("etac δ² : chrom=2 ", getterm(t2.etac, delta = 2), "  chrom=3 ", getterm(t3.etac, delta = 2),
            "  independent: tracked -3 z₃/C = ", -3z3 / 2)
    println("alphac δ²: chrom=2 ", getterm(t2.alphac, delta = 2), "  chrom=3 ", getterm(t3.alphac, delta = 2))

    # First-order perturbation theory: a thin octupole adds dQ1/dJ1 = K3L βx²/(16π), compared with the
    # increment at order 2 and at the complete order 3.
    K3 = 1000.0
    el = Species("electron")
    base = Beamline([qf, d, qd, d], species_ref = el, pc_ref = 1e9)
    oct = Beamline([Multipole(Kn3L = K3), qf, d, qd, d], species_ref = el, pc_ref = 1e9)
    βx = twiss(base, rf_on = false, at = [1], cols = ["beta1"]).beta1[1]
    inc(kw) = getterm(twiss(oct; rf_on = false, at = [], kw...).q1, J1 = 1) - getterm(twiss(base; rf_on = false, at = [], kw...).q1, J1 = 1)
    println("dQ1/dJ1 octupole increment: chrom=2 ", inc((chrom = 2,)), "  chrom=3 ", inc((chrom = 3,)), "  order=2 ", inc((order = 2,)),
            "  order=3 ", inc((order = 3,)), "  analytic K3L βx²/(16π) = ", K3 * βx^2 / (16pi))

    # Bunched (RF on, order = chrom): J3 and cross terms at order 2 against the complete order 3.
    rfc = RFCavity(L = 0.5, voltage = 2e6, harmon = 1, zero_phase = PhaseRef.BelowTransition)
    rbl = Beamline([Multipole(Kn3L = K3), qf, d, qd, d, rfc], species_ref = Species("proton"), pc_ref = 1e9)
    for o in (2, 3)
        tw = twiss(rbl; order = o, at = [])
        println("bunched order=$o: q1 J1 ", getterm(tw.q1, J1 = 1), "  q1 J3 ", getterm(tw.q1, J3 = 1), "  q3 J1 ", getterm(tw.q3, J1 = 1),
                "  q3 J3 ", getterm(tw.q3, J3 = 1), "  etac J3 ", getterm(tw.etac, J3 = 1))
    end

    # Guard comparison: the requested monomial (current guard) and the map term it is read from
    # (request + x for q1, request + δ for the coasting etac).
    valid(desc, m) = GTPSA.mad_desc_isvalidm(desc.desc, Cint(6), UInt8.(m))
    for (lab, kw, what, m, e) in (("chrom=2", (chrom = 2,), "q1 δ²", [0, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 0]),
                                  ("chrom=2", (chrom = 2,), "q1 δ", [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]),
                                  ("chrom=2", (chrom = 2,), "etac δ²", [0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 1]),
                                  ("chrom=3", (chrom = 3,), "q1 J1", [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
                                  ("chrom=3", (chrom = 3,), "q1 δ²", [0, 0, 0, 0, 0, 2], [1, 0, 0, 0, 0, 0]),
                                  ("order=2", (order = 2,), "q1 J1", [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
                                  ("order=3", (order = 3,), "q1 J1", [1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]))
        desc = GTPSA.getdesc(twiss(base; rf_on = false, at = [], kw...).q1.taylor_series)
        println("guard $lab $what: request valid = ", valid(desc, m), "  map term valid = ", valid(desc, m .+ e))
    end

    bad = getterm(t2.q1, delta = 2) == 0.0 && abs(c2) > 0.1 && getterm(t2.etac, delta = 2) == 0.0 && inc((order = 2,)) == 0.0
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | xhigh | 2026-10-23 | SciBmad main (`5f0fb83ac889`), pinned and main-branch sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-23 | SciBmad main (`5f0fb83ac889`), pinned and main-branch sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | SciBmad 0.6.0 (`5f0fb83ac889`), source (SciBmad 5f0fb83ac889, NonlinearNormalForm 7e4053985736, GTPSA a5790a7deaa6, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, Float64, one thread |

