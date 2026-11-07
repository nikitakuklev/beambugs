---
title: "LAT-01: Rigidity sign \"correction\" keeps a negative rigidity for positive species"
hide:
  - toc
---

# LAT-01 — Rigidity sign "correction" keeps a negative rigidity for positive species

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on Beamlines release [v0.11.0](https://github.com/bmad-sim/Beamlines.jl/tree/v0.11.0) (`35fe016`) |
| Source | [`Beamlines.jl/src/virtual.jl:629-638`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L629-L638)<br>[`Beamlines.jl/src/virtual.jl:641-644`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L641-L644)<br>[`Beamlines.jl/src/utils.jl:5`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/utils.jl#L5)<br>[`:632`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L632)<br>[`:637`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/virtual.jl#L637) |

Giving a positive species a negative reference rigidity `p_over_q_ref`, in the `Beamline` constructor, by assignment or by switching `species_ref` from electron to proton, logs a sign correction but keeps the negative value. Signed rigidity R obeys pc = qRc with a forward reference momentum, so R must carry the sign of the charge q; the same state requested through `pc_ref` is rejected. For a proton line given `p_over_q_ref` = −3, the stored reference is R = −3.000 T·m, `pc_ref` = −0.8994 GeV and `E_ref` = −1.300 GeV, instead of +3.000 T·m, +0.8994 GeV and +1.300 GeV. Negatively charged species are corrected; for positive species every absolute reference quantity derived from the rigidity has the wrong sign.

Run `julia LAT-01.jl`.

??? example "Reproducer: LAT-01.jl"

    [Download LAT-01.jl](LAT-01.jl)

    ```julia
    # LAT-01: Rigidity sign "correction" keeps a negative rigidity for positive species
    #
    # Giving a positive species a negative reference rigidity `p_over_q_ref`, in the `Beamline`
    # constructor, by assignment or by switching `species_ref` from electron to proton, logs a sign
    # correction but keeps the negative value. Signed rigidity R obeys pc = qRc with a forward reference
    # momentum, so R must carry the sign of the charge q; the same state requested through `pc_ref` is
    # rejected. For a proton line given `p_over_q_ref` = −3, the stored reference is R = −3.000 T·m,
    # `pc_ref` = −0.8994 GeV and `E_ref` = −1.300 GeV, instead of +3.000 T·m, +0.8994 GeV and +1.300
    # GeV. Negatively charged species are corrected; for positive species every absolute reference
    # quantity derived from the rigidity has the wrong sign.
    #
    # The script sets `p_over_q_ref` = −3 on a proton `Beamline` three ways: in the constructor, through
    # an electron-to-proton species change and by assignment to a proton line. It compares the stored R,
    # `pc_ref` and `E_ref` with the forward-momentum values R = +3 T·m, pc = 3c eV and E = √(pc² + m²).
    # An electron line given R = +3, which the setter correctly flips to −3, is the control.
    using SciBmad
    p = Species("proton")
    c = C_LIGHT
    m = massof(p)

    # Expected: forward reference momentum pc_ref >= 0 (negative pc_ref/E_ref is rejected, virtual.jl:641-644),
    # pc = q*R*c (Beamlines src/utils.jl:5), so R must carry sign(q) = +1 for a proton: R = +3 T m,
    # pc = 3c eV, E = sqrt(pc^2 + m^2).
    R_exp = 3.0
    pc_exp = 3.0 * c
    E_exp = sqrt(pc_exp^2 + m^2)

    cases = Dict{String,Any}()
    # (a) Beamline constructor: p_over_q_ref set first, then species_ref (beamline.jl:250-259)
    bl = Beamline([Marker()], species_ref=p, p_over_q_ref=-3.0)
    cases["ctor proton R=-3"] = (bl.p_over_q_ref, bl.pc_ref, bl.E_ref)
    # (b) species change electron -> proton on a stored rigidity
    bl2 = Beamline([Marker()], species_ref=Species("electron"), p_over_q_ref=-3.0)
    bl2.species_ref = p
    cases["electron->proton"] = (bl2.p_over_q_ref, bl2.pc_ref, bl2.E_ref)
    # (c) setting a negative rigidity with a stored proton species
    bl3 = Beamline([Marker()], species_ref=p, pc_ref=1e9)
    bl3.p_over_q_ref = -3.0
    cases["set R=-3 on proton"] = (bl3.p_over_q_ref, bl3.pc_ref, bl3.E_ref)
    # control: negative-charge branch is corrected
    ble = Beamline([Marker()], species_ref=Species("electron"), p_over_q_ref=3.0)
    println("control electron R=+3 -> R = ", ble.p_over_q_ref, " (expected -3.0)")

    bad = false
    for (k, (R, pc, E)) in sort(collect(cases))
        println(rpad(k, 20), " observed R = ", R, ", pc_ref = ", pc, ", E_ref = ", E)
        global bad |= !(R == R_exp && isapprox(pc, pc_exp; rtol=1e-15) && isapprox(E, E_exp; rtol=1e-15))
    end
    println("expected             R = ", R_exp, ", pc_ref = ", pc_exp, ", E_ref = ", E_exp)
    println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | Beamlines 0.11.0 (`35fe0162c72d`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

