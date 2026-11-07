---
title: "OPT-17: Orbital caps of 1 drop the start-point dispersion feed-down (default twiss(chrom=2), documented knob descriptors)"
hide:
  - toc
---

# OPT-17 — Orbital caps of 1 drop the start-point dispersion feed-down (default `twiss(chrom=2)`, documented knob descriptors)

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P0 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad commit [`ae9423d`](https://github.com/bmad-sim/SciBmad.jl/commit/ae9423d703b4), committed 2026-10-28 |
| Source | [`src/twiss/twiss.jl:283-286`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L283-L286)<br>[`twiss.jl:82`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L82)<br>[`docs/src/optimize.md:130`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/optimize.md#L130)<br>[`140`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/optimize.md#L140)<br>[`147`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/optimize.md#L147)<br>[`docs/src/optimize.md:100`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/optimize.md#L100)<br>[`63`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/parametric-nf.md#L63)<br>[`130`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/parametric-nf.md#L130)<br>[`167`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/docs/src/parametric-nf.md#L167)<br>[`NonlinearNormalForm.jl/src/normal.jl:61-94`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L61-L94)<br>[`normal.jl:116`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L116)<br>[`twiss.jl:78-85`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L78-L85)<br>[`:102`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L102) |

With `twiss(chrom=2)`, the default GTPSA descriptor caps each phase-space coordinate at degree 1 and so drops the squares x² and pₓ² that carry the dispersion feed-down, silently corrupting chromatic optics. Chromaticity is dQ/dδ of the one-turn map linearized on the momentum-dependent closed orbit; tracking and `twiss(order=2, chrom=2)` agree on it. In the documentation's 18 GeV/c electron FODO with sextupoles, started where the dispersion is 0.5317 m, ξₓ is +0.1220 instead of −0.06914 and `dbeta1` is −39.28 m instead of −30.65 m, and the documented matching recipe reaches a tracked ξₓ of 0.5128 instead of 1. Dispersive starts with feed-down from bends or multipoles, and the documented low-cap knob descriptors for orbit-moving knobs without dispersion, show the error; on-momentum linear optics stay correct.

Run `julia OPT-17.jl`.

??? example "Reproducer: OPT-17.jl"

    [Download OPT-17.jl](OPT-17.jl)

    ```julia
    # OPT-17: Orbital caps of 1 drop the start-point dispersion feed-down (default `twiss(chrom=2)`,
    # documented knob descriptors)
    #
    # With `twiss(chrom=2)`, the default GTPSA descriptor caps each phase-space coordinate at degree 1
    # and so drops the squares x² and pₓ² that carry the dispersion feed-down, silently corrupting
    # chromatic optics. Chromaticity is dQ/dδ of the one-turn map linearized on the momentum-dependent
    # closed orbit; tracking and `twiss(order=2, chrom=2)` agree on it. In the documentation's 18 GeV/c
    # electron FODO with sextupoles, started where the dispersion is 0.5317 m, ξₓ is +0.1220 instead of
    # −0.06914 and `dbeta1` is −39.28 m instead of −30.65 m, and the documented matching recipe reaches
    # a tracked ξₓ of 0.5128 instead of 1. Dispersive starts with feed-down from bends or multipoles,
    # and the documented low-cap knob descriptors for orbit-moving knobs without dispersion, show the
    # error; on-momentum linear optics stay correct.
    #
    # The script builds the documentation's FODO cell with sextupoles and bends (18 GeV/c electrons, RF
    # off), started at the focusing quadrupole, and takes the expected ξ, `dbeta1` and `dx_2` from
    # tracking: the closed orbit at each δ, the ForwardDiff one-turn Jacobian there, and five-point
    # differences in δ. It compares the default `twiss(chrom=2)` and `twiss(order=2, chrom=2)` with
    # these, shows the truncation directly on (x + pₓ)² and with caps that restore x² alone or both
    # squares, and compares the chromaticity Jacobian with respect to the two sextupole strengths from
    # the documented knob descriptor with tracking.
    using SciBmad, LinearAlgebra
    const FD = SciBmad.ForwardDiff
    mk(ksf, ksd) = Beamline([Quadrupole(Kn1=0.36, L=0.5), Sextupole(Kn2=ksf, L=0.2), Drift(L=0.2), SBend(L=6.5, angle=pi / 132),
                             Drift(L=0.2), Quadrupole(Kn1=-0.36, L=0.5), Sextupole(Kn2=ksd, L=0.2), Drift(L=0.2),
                             SBend(L=6.5, angle=pi / 132), Drift(L=0.2)], species_ref=Species("electron"), pc_ref=18e9)
    # Expected values from tracking: closed orbit at δ, ForwardDiff one-turn Jacobian there, tune and β
    # of its 2×2 blocks.
    function trk(bl, δ)
        v = find_closed_orbit(bl; v0=[0.0 0 0 0 0 δ], rf_on=false).v0
        J = FD.jacobian(u -> vec(track(bl, v0=reshape(u, 1, 6), rf_on=false).v[1, :, end]), vec(v))
        tb(m) = (c = tr(m) / 2; s = sign(m[1, 2]) * sqrt(1 - c^2); (mod(atan(s, c) / 2pi, 1), m[1, 2] / s))
        qx, bx = tb(J[1:2, 1:2]); qy, _ = tb(J[3:4, 3:4])
        (qx=qx, qy=qy, bx=bx, x=v[1])
    end
    d1(f, h) = (-f(2h) + 8f(h) - 8f(-h) + f(-2h)) / (12h)
    d2(f, h) = (-f(2h) + 16f(h) - 30f(0.0) + 16f(-h) - f(-2h)) / (12h^2)
    xi(bl, h=1e-4) = (d1(δ -> trk(bl, δ).qx, h), d1(δ -> trk(bl, δ).qy, h))
    bl = mk(1.2, -1.2)
    xr = xi(bl); xr2 = xi(bl, 2e-4)
    dbr = d1(δ -> trk(bl, δ).bx, 1e-4); dxr = d1(δ -> trk(bl, δ).x, 1e-4); dx2r = d2(δ -> trk(bl, δ).x, 1e-3)
    function tw(bl; kw...)
        t = twiss(bl; rf_on=false, cols=["beta1", "dbeta1", "dx", "dx_2"], as_taylor_series=true, kw...)
        (xi1=getterm(t.q1, delta=1), xi2=getterm(t.q2, delta=1), b=scalar(t.beta1[1]), db=scalar(t.dbeta1[1]),
         dbend=scalar(t.dbeta1[end]), dx=scalar(t.dx[1]), dx2=scalar(t.dx_2[1]))   # Taylor path: dx_2 = d2x/ddelta^2
    end
    # The default descriptor is Descriptor([order×5, chrom], max(order, chrom)), so twiss(chrom=2) caps
    # every orbital coordinate at degree 1 and GTPSA drops every monomial containing u_i². With M_k,ij
    # the coefficient of u_i u_j in row k, the linear part about the δ-dependent fixed point u* = Dδ is
    # ∂M_k/∂u_j = M_kj + δ [M_k,jδ + Σ_{i≠j} M_k,ij D_i + 2 M_k,jj D_j] + O(δ²); the 2 M_k,jj D_j terms
    # come from the dropped squares, which sextupoles and bend curvature make nonzero.
    t1 = tw(bl; chrom=2); t2 = tw(bl; order=2, chrom=2)
    println("tracking: D_x = ", dxr, " m at the start; xi = ", xr, " (h=2e-4: ", xr2, ")")
    println("tracking: dbeta1 = ", dbr, "  dx_2 = ", dx2r)
    println("twiss(chrom=2)          : xi1 = ", t1.xi1, " xi2 = ", t1.xi2, " dx = ", t1.dx, " dbeta1 = ", t1.db, " (END row ", t1.dbend, ") dx_2 = ", t1.dx2)
    println("independent: twiss(order=2, chrom=2): xi1 = ", t2.xi1, " xi2 = ", t2.xi2, " dx = ", t2.dx, " dbeta1 = ", t2.db, " (END row ", t2.dbend, ") dx_2 = ", t2.dx2)
    # Mechanism: caps of 1 keep x*px but drop x^2 and px^2.
    d = Descriptor([1, 1, 1, 1, 1, 2], 2); x = vars(d)
    sq = (x[1] + x[2])^2
    println("independent: under caps [1,1,1,1,1,2], (x+px)^2 - 2*x*px = ", normTPS(sq - 2 * x[1] * x[2]))
    t3 = tw(bl; GTPSA_descriptor=Descriptor([2, 2, 1, 1, 1, 2], 2)); t4 = tw(bl; GTPSA_descriptor=Descriptor([2, 1, 1, 1, 1, 2], 2))
    println("independent: caps [2,2,1,1,1,2] xi1 = ", t3.xi1, "   caps [2,1,1,1,1,2] (px^2 still cut) xi1 = ", t4.xi1)
    # Knob Jacobian d(xi)/d(ksf, ksd) with the documented descriptor, against tracking.
    kbl = mk(DefExpr(c -> c.ksf), DefExpr(c -> c.ksd))
    function chromjac(desc)
        dk = params(desc); kbl.context.ksf = 1.2 + dk[1]; kbl.context.ksd = -1.2 + dk[2]
        t = twiss(kbl, at=[], GTPSA_descriptor=desc, rf_on=false)
        scalarize!(kbl); kbl.context.ksf = 1.2; kbl.context.ksd = -1.2
        jac([getterm(t.q1, delta=1, as_taylor_series=true), getterm(t.q2, delta=1, as_taylor_series=true)])
    end
    hk = 0.1
    jref = hcat([(collect(xi(mk(1.2 + (j == 1) * hk, -1.2 + (j == 2) * hk))) .- collect(xi(mk(1.2 - (j == 1) * hk, -1.2 - (j == 2) * hk)))) ./ (2hk) for j in 1:2]...)
    jdoc = chromjac(Descriptor([1, 1, 1, 1, 1, 2], 3, [1, 1], 1)); jfix = chromjac(Descriptor([2, 2, 2, 2, 2, 2], 3, [1, 1], 1))
    println("chromaticity Jacobian, documented Descriptor([1,1,1,1,1,2],3,[1,1],1): ", jdoc)
    println("chromaticity Jacobian, tracking (central difference, dk = 0.1; xi is linear in k2): ", jref)
    println("independent: Descriptor([2,2,2,2,2,2],3,[1,1],1): ", jfix, "  max|diff to tracking| = ", maximum(abs.(jfix - jref)))
    bad = abs(t1.xi1 - xr[1]) > 1e-9 && abs(t2.xi1 - xr[1]) < 1e-9 && abs(t1.db - dbr) > 1e-6 && maximum(abs.(jdoc - jref)) > 1e-6
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), release sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | SciBmad 0.6.0 (`5f0fb83ac889`), source (SciBmad 5f0fb83ac889, NonlinearNormalForm 7e4053985736, GTPSA a5790a7deaa6, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, Float64, one thread |

