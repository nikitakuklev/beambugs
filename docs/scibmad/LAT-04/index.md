---
title: "LAT-04: Harmonic RF frequency uses branch-global s, not the containing Beamline length"
hide:
  - toc
---

# LAT-04 — Harmonic RF frequency uses branch-global `s`, not the containing Beamline length

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on Beamlines main (`35fe016`, head build) |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`Beamlines.jl/src/virtual.jl:556`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L556)<br>[`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/utils_bl.jl:204`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/ext/BeamTrackingBeamlinesExt/utils_bl.jl#L204)<br>[`Beamlines.jl/src/beamline.jl:557-566`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/beamline.jl#L557-L566)<br>[`Beamlines.jl/src/rf.jl:37-38`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/rf.jl#L37-L38)<br>[`Beamlines.jl/src/branch.jl:156-158`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/branch.jl#L156-L158) |

A harmonic-number cavity's `rf_frequency` getter and its tracking setup take as circumference the branch-global end position `s` of the cavity's Beamline, which includes every upstream Beamline of the `Branch`. The documentation defines `harmon` relative to the containing Beamline's own length, and the same cavity Beamline standing alone gives exactly that frequency. For a 1 GeV/c proton in a 3 m cavity Beamline with `harmon` = 1, adding a 2 m upstream Beamline changes the frequency from 72.88 MHz to 43.73 MHz, and a 4 m one gives 31.23 MHz. Adding a 7 m downstream Beamline leaves 43.73 MHz, although the whole-Branch harmonic is 18.22 MHz. Cavities after upstream Beamlines therefore violate the documented length contract.

Run `julia LAT-04.jl`.

??? example "Reproducer: LAT-04.jl"

    [Download LAT-04.jl](LAT-04.jl)

    ```julia
    # LAT-04: Harmonic RF frequency uses branch-global `s`, not the containing Beamline length
    #
    # A harmonic-number cavity's `rf_frequency` getter and its tracking setup take as circumference the
    # branch-global end position `s` of the cavity's Beamline, which includes every upstream Beamline of
    # the `Branch`. The documentation defines `harmon` relative to the containing Beamline's own length,
    # and the same cavity Beamline standing alone gives exactly that frequency. For a 1 GeV/c proton in
    # a 3 m cavity Beamline with `harmon` = 1, adding a 2 m upstream Beamline changes the frequency from
    # 72.88 MHz to 43.73 MHz, and a 4 m one gives 31.23 MHz. Adding a 7 m downstream Beamline leaves
    # 43.73 MHz, although the whole-Branch harmonic is 18.22 MHz. Cavities after upstream Beamlines
    # therefore violate the documented length contract.
    #
    # The script places a 3 m `RFCavity` Beamline (`harmon` = 1, 1 GeV/c protons) after a 2 m or 4 m
    # drift Beamline in a `Branch`, once also with a 7 m drift Beamline downstream. For each, it
    # compares the `rf_frequency` getter and the frequency of the tracking setup with the documented
    # v/(3 m); the whole-Branch harmonic v/(12 m) is printed for reference.
    using SciBmad
    const BT = SciBmad.BeamTracking
    ext = Base.get_extension(BT, :BeamTrackingBeamlinesExt)
    p = Species("proton"); pc = 1e9
    v = C_LIGHT * pc / sqrt(pc^2 + massof(p)^2)
    # harmon is documented relative to the length of the containing Beamline, so f = h v/(3 m); the getter
    # and the tracking setup divide by the cavity's s_downstream, which adds all upstream Beamlines of the Branch.
    function cav_freq(Lup; Ldown=nothing)
        a = Beamline([Drift(L=Lup)], species_ref=p, pc_ref=pc)
        b = Beamline([RFCavity(L=3.0, harmon=1, voltage=1.0)])
        parts = isnothing(Ldown) ? [a, b] : [a, b, Beamline([Drift(L=Ldown)])]
        br = Branch(parts)
        e = br.beamlines[2].line[end]
        return e.rf_frequency, ext.rf_omega_calc(e.RFParams, e.BeamlineParams) / (2pi), e.s_downstream
    end
    bad = false
    for (Lup, Ldown) in ((2.0, nothing), (4.0, nothing), (2.0, 7.0))
        f, fw, s = cav_freq(Lup; Ldown)
        println("upstream L = $Lup, downstream L = $Ldown: getter f = $f Hz, tracking f = $fw Hz, s_downstream = $s")
        global bad |= !isapprox(f, v / 3; rtol=1e-14) || !isapprox(fw, v / 3; rtol=1e-14)
    end
    println("expected (containing Beamline length 3 m): f = v/3 = ", v / 3, " Hz")
    println("        (whole Branch 2+3+7 m, for reference): v/12 = ", v / 12, " Hz")
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

