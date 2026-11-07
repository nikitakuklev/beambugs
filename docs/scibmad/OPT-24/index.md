---
title: "OPT-24: twiss overwrites the caller's v0 with the closed orbit"
hide:
  - toc
---

# OPT-24 — `twiss` overwrites the caller's `v0` with the closed orbit

| | |
|---|---|
| Code | [SciBmad](../index.md) · Optics and maps |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-24 on SciBmad commit [`5f0fb83`](https://github.com/bmad-sim/SciBmad.jl/commit/5f0fb83ac889), committed 2026-10-15 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`src/twiss/twiss.jl:248`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L248)<br>[`:364-369`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L364-L369)<br>[`closed_orbit.jl:241`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/closed_orbit.jl#L241)<br>[`:235`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/closed_orbit.jl#L235)<br>[`:132`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/closed_orbit.jl#L132)<br>[`:143`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/closed_orbit.jl#L143)<br>[`twiss.jl:115-116`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/twiss/twiss.jl#L115-L116) |

`twiss(bl; v0=guess)` hands the caller's matrix to the in-place closed-orbit solver, which overwrites it with the closed orbit, although the `twiss` docstring calls `v0` an initial guess and does not mention the overwrite. `find_closed_orbit` documents this mutation; documenting the overwrite in `twiss` or removing it settles the contract question. In a coasting thin FODO with a 10⁻⁴ rad kicker and 1 GeV electrons, a guess at momentum deviation δ = 0.01 becomes x = −3.570×10⁻⁴ m and pₓ = −1.928×10⁻⁴, and tracking confirms a transverse fixed-point residual of 5.421×10⁻²⁰. Reusing the matrix changes later tracking launches or guesses; periodic RF calls also mutate it, while calls with `a_initial` bypass the solver.

Run `julia OPT-24.jl`.

??? example "Reproducer: OPT-24.jl"

    [Download OPT-24.jl](OPT-24.jl)

    ```julia
    # OPT-24: `twiss` overwrites the caller's `v0` with the closed orbit
    #
    # `twiss(bl; v0=guess)` hands the caller's matrix to the in-place closed-orbit solver, which
    # overwrites it with the closed orbit, although the `twiss` docstring calls `v0` an initial guess
    # and does not mention the overwrite. `find_closed_orbit` documents this mutation; documenting the
    # overwrite in `twiss` or removing it settles the contract question. In a coasting thin FODO with a
    # 10⁻⁴ rad kicker and 1 GeV electrons, a guess at momentum deviation δ = 0.01 becomes x =
    # −3.570×10⁻⁴ m and pₓ = −1.928×10⁻⁴, and tracking confirms a transverse fixed-point residual of
    # 5.421×10⁻²⁰. Reusing the matrix changes later tracking launches or guesses; periodic RF calls also
    # mutate it, while calls with `a_initial` bypass the solver.
    #
    # The script calls `twiss` with an explicit guess on that cell and on the same cell with an RF
    # cavity (1 GeV/c protons, RF on) and prints the guess before and after. It confirms that the
    # overwritten matrix is the closed orbit by tracking one turn from it (all coordinates but z for the
    # coasting beam, all six with RF) and by calling `find_closed_orbit` on a copy of the original
    # guess. A last call with an integer guess shows that the solve works in the caller's element type.
    using SciBmad
    qf = Multipole(Kn1L=0.8); qd = Multipole(Kn1L=-0.7); d = Drift(L=1.0); k = Multipole(Kn0L=1e-4)
    rfc = RFCavity(L=0.5, voltage=2e6, harmon=1, zero_phase=PhaseRef.BelowTransition)
    cases = (("coasting electrons", Beamline([qf, d, qd, d, k], species_ref=Species("electron"), pc_ref=1e9), false, [0.0 0 0 0 0 0.01]),
             ("protons with RF", Beamline([qf, d, qd, d, k, rfc], species_ref=Species("proton"), pc_ref=1e9), true, [0.0 0 0 0 0 0]))
    bad = 0
    for (name, bl, rf, guess) in cases
      v0 = copy(guess)
      twiss(bl; rf_on=rf, v0=v0, cols=["beta1"])
      println(name, ": v0 before twiss = ", guess, ", after = ", v0)
      turn = track(bl; v0=copy(v0), rf_on=rf).v[1, :, end]
      closed = rf ? (1:6) : [1, 2, 3, 4, 6] # a coasting beam's z advances every turn
      co = find_closed_orbit(bl; v0=copy(guess), rf_on=rf).v0
      println("  independent: |track(v0 after) - v0 after| = ", maximum(abs.(turn[closed] .- v0[closed])),
              ", |find_closed_orbit(guess) - v0 after| = ", maximum(abs.(co .- v0)))
      global bad += v0 != guess
    end
    # The solve also runs in the caller's element type: an integer guess cannot hold the orbit.
    e = try twiss(cases[1][2]; rf_on=false, v0=[0 0 0 0 0 0]); "no error" catch err; err end
    println("Int v0: ", e isa Exception ? sprint(showerror, e) : e)
    println(bad == length(cases) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-23 | SciBmad main (`5f0fb83ac889`), pinned and main-branch sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `claude-opus-5-5` | high | 2026-10-24 | SciBmad main (`5f0fb83ac889`), pinned and main-branch sources, Julia 1.13.2, CPU, Float64 |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | SciBmad 0.6.0 (`5f0fb83ac889`), source (SciBmad 5f0fb83ac889, NonlinearNormalForm 7e4053985736, GTPSA a5790a7deaa6, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, Float64, one thread |

