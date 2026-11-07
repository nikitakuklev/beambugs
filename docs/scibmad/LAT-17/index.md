---
title: "LAT-17: ags-spin-ramp.ipynb Python example calls the removed rotate_spins"
hide:
  - toc
---

# LAT-17 — `ags-spin-ramp.ipynb` Python example calls the removed `rotate_spins`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad commit [`ae9423d`](https://github.com/bmad-sim/SciBmad.jl/commit/ae9423d703b4), committed 2026-10-28 |
| Source | [`SciBmad.jl/examples/python/ags-spin-ramp.ipynb:272`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/examples/python/ags-spin-ramp.ipynb#L272)<br>[`:400`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/examples/python/ags-spin-ramp.ipynb#L400)<br>[`:234`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/examples/python/ags-spin-ramp.ipynb#L234)<br>[`SciBmad.jl/src/track.jl:311`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L311) |

The Python example notebook `ags-spin-ramp.ipynb` rotates spins with `jl.rotate_spins(b0.coords.q, s_initial)`, and its text also names `rotate_spins`, but SciBmad defines neither `rotate_spins` nor `rotate_spins!`. The spin-rotation function is `track_spin(q, s0)`, which takes the N×4×n_saved quaternion array of a `TrackingResult`; the repository history shows `rotate_spins` deleted and the Julia version of the same notebook moved to `track_spin`. The Python example therefore fails at its first spin step. The failure is loud.

Run `julia LAT-17.jl`.

??? example "Reproducer: LAT-17.jl"

    [Download LAT-17.jl](LAT-17.jl)

    ```julia
    # LAT-17: `ags-spin-ramp.ipynb` Python example calls the removed `rotate_spins`
    #
    # The Python example notebook `ags-spin-ramp.ipynb` rotates spins with `jl.rotate_spins(b0.coords.q,
    # s_initial)`, and its text also names `rotate_spins`, but SciBmad defines neither `rotate_spins`
    # nor `rotate_spins!`. The spin-rotation function is `track_spin(q, s0)`, which takes the
    # N×4×n_saved quaternion array of a `TrackingResult`; the repository history shows `rotate_spins`
    # deleted and the Julia version of the same notebook moved to `track_spin`. The Python example
    # therefore fails at its first spin step. The failure is loud.
    #
    # Setup: `SCIBMAD_SOURCE` set to a clone of bmad-sim/SciBmad.jl at the pinned commit `5f0fb83`; the
    # script reads an example notebook from the clone.
    #
    # The script prints the notebook lines that call `jl.rotate_spins` and checks whether SciBmad
    # defines `rotate_spins`, `rotate_spins!`, `track_spin` and `track_spin!`; the example's call target
    # should exist.
    using SciBmad
    haskey(ENV, "SCIBMAD_SOURCE") || error("set SCIBMAD_SOURCE to a SciBmad.jl checkout (commit 5f0fb83 or the build under test)")
    nb = joinpath(ENV["SCIBMAD_SOURCE"], "examples", "python", "ags-spin-ramp.ipynb")
    isfile(nb) || error("notebook not found at $nb")
    for (i, l) in enumerate(readlines(nb))
        occursin("jl.rotate_spins", l) && println("notebook line $i: ", strip(l))
    end
    has = [s => isdefined(SciBmad, s) for s in (:rotate_spins, :rotate_spins!, :track_spin, :track_spin!)]
    println("defined in SciBmad: ", has)
    println("track_spin methods: ", methods(track_spin))
    println("expected: the example's call target exists")
    println(isdefined(SciBmad, :rotate_spins) ? "NOT REPRODUCED" : "DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

