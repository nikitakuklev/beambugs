---
title: "OPT-12: Spin normal form gives NaN for an exactly vertical (or absent) spin rotation"
hide:
  - toc
---

# OPT-12 — Spin normal form gives NaN for an exactly vertical (or absent) spin rotation

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-24 on NonlinearNormalForm release [v0.5.3](https://github.com/bmad-sim/NonlinearNormalForm.jl/tree/v0.5.3) (`7e40539`) |
| Source | [`NonlinearNormalForm.jl/src/normal.jl:219-224`](https://github.com/bmad-sim/NonlinearNormalForm.jl/blob/7e40539857368faf9784736ad29c8f0ae6e70a23/src/normal.jl#L219-L224)<br>[`forest/code/Ci_tpsa.f90:13369-13383`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/forest/code/Ci_tpsa.f90#L13369-L13383) |

NonlinearNormalForm's spin normal form throws `InexactError: Float64(NaN + NaN*im)` when the closed-orbit spin axis is exactly vertical or there is no spin rotation. The step that aligns the axis with ŷ divides by a zero axis norm or transverse axis length, although these maps admit a normal form and Forest's FPP handles them explicitly. For a synthetic map that rotates spin by 0.6 rad about ŷ, tilting the axis by 1e-9 rad returns the correct spin-tune magnitude 0.09549 but leaves a transverse quaternion component of 2.96e-10, because the alignment angle loses the tilt. The public `twiss(spin=true)` fails in checked planar rings and straight cells, including RF and coasting cases; the thick-bend quickstart ring escapes only through transverse round-off of order 1e-34.

Run `julia OPT-12.jl`.

??? example "Reproducer: OPT-12.jl"

    [Download OPT-12.jl](OPT-12.jl)

    ```julia
    # OPT-12: Spin normal form gives NaN for an exactly vertical (or absent) spin rotation
    #
    # NonlinearNormalForm's spin normal form throws `InexactError: Float64(NaN + NaN*im)` when the
    # closed-orbit spin axis is exactly vertical or there is no spin rotation. The step that aligns the
    # axis with ŷ divides by a zero axis norm or transverse axis length, although these maps admit a
    # normal form and Forest's FPP handles them explicitly. For a synthetic map that rotates spin by 0.6
    # rad about ŷ, tilting the axis by 1e-9 rad returns the correct spin-tune magnitude 0.09549 but
    # leaves a transverse quaternion component of 2.96e-10, because the alignment angle loses the tilt.
    # The public `twiss(spin=true)` fails in checked planar rings and straight cells, including RF and
    # coasting cases; the thick-bend quickstart ring escapes only through transverse round-off of order
    # 1e-34.
    #
    # The script builds a stable uncoupled `Descriptor(6, 1)` map whose spin quaternion (cos θ, 0, sin
    # θ, 0), θ = 0.3, rotates spin by 2θ about ŷ, and computes its spin tune with the axis exactly
    # vertical and tilted by 1e-9 rad; the expected magnitude is 2θ/2π. It then runs `twiss(rf_on=false,
    # spin=true)` for 1 GeV/c electrons on a cell of thin quadrupoles whose only bend is a thin
    # vertical-field kick, and on a straight cell, printing the tracked closed-orbit quaternion and the
    # error.
    using SciBmad, LinearAlgebra
    const TI = SciBmad.TI
    d = Descriptor(6, 1); init = TI.InitGTPSA{GTPSA.Dynamic,Descriptor}(dynamic_descriptor = d)
    R(q) = [cospi(2q) sinpi(2q); -sinpi(2q) cospi(2q)]
    A = zeros(6, 6); for (j, q) in enumerate([0.21, 0.31, 0.07]); A[2j-1:2j, 2j-1:2j] = R(q); end
    θ = 0.3
    spinmap(ε) = DAMap(init = init, nv = 6, v_matrix = A, q = [cos(θ), sin(θ) * sin(ε), sin(θ) * cos(ε), 0.0])
    res(f) = try f() catch err; "ERROR " * first(split(sprint(showerror, err), '\n')) end
    # The normal form rotates n0 onto ŷ with qr = (cos α/2, sin α/2 · n0z/ρ, 0, −sin α/2 · n0x/ρ), ρ =
    # √(n0x² + n0z²). For n0 = ±ŷ, the generic case of a planar ring, ρ = 0 and the rotation is 0/0 =
    # NaN, although the map is already aligned.
    r0 = res(() -> SciBmad._a_r_tunes(spinmap(0.0))[3][4])
    r1 = res(() -> SciBmad._a_r_tunes(spinmap(1e-9))[3][4])
    println("axis exactly vertical: spin tune -> ", r0)
    println("axis tilted 1e-9 rad : spin tune -> ", r1 isa String ? r1 : scalar(r1), "   expected |ν| = ", 2θ / 2pi)
    # Public path: twiss(spin=true) on a ring whose only bending is a thin vertical-field kick
    # (tracked closed-orbit quaternion has exactly zero x/z components), and on a straight cell
    # (identity spin map). Quickstart's thick SBend ring survives only because its tracked
    # quaternion carries ~1e-34 round-off in q1/q3.
    qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); dr = Drift(L = 1.0)
    for (name, line) in (("thin dipole kick", [qf, dr, qd, dr, Multipole(Kn0L = 1e-3)]), ("straight", [qf, dr, qd, dr]))
        bl = Beamline(line, species_ref = Species("electron"), pc_ref = 1e9)
        q = track(bl, v0 = zeros(1, 6), spin = true).q[1, :, end]
        println("twiss(spin=true), ", name, ": closed-orbit q = ", q, " -> ", res(() -> twiss(bl, rf_on = false, spin = true).qspin))
    end
    println(r0 isa String ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), release sources, Julia 1.13.2, CPU, Float64; SciBmad 0.6.0 (62399f930847) for the twiss path |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), release sources, Julia 1.13.2, CPU, Float64; SciBmad 0.6.0 (62399f930847) for the twiss path |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | NonlinearNormalForm 0.5.3 (`7e4053985736`), release sources, Julia 1.13.2, CPU, Float64; SciBmad 0.6.0 (62399f930847) for the twiss path |

