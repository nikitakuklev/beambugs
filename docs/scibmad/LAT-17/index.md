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
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad main (`ae9423d`) |
| Source | [`SciBmad.jl/examples/python/ags-spin-ramp.ipynb:272`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/python/ags-spin-ramp.ipynb#L272)<br>[`:400`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/python/ags-spin-ramp.ipynb#L400)<br>[`:234`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/examples/python/ags-spin-ramp.ipynb#L234)<br>[`SciBmad.jl/src/track.jl:311`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/src/track.jl#L311) |

The Python example notebook `ags-spin-ramp.ipynb` rotates spins with `jl.rotate_spins(b0.coords.q, s_initial)`, and its text also names `rotate_spins`, but SciBmad defines neither `rotate_spins` nor `rotate_spins!`. The spin-rotation function is `track_spin(q, s0)`, which takes the N×4×n_saved quaternion array of a `TrackingResult`; the repository history shows `rotate_spins` deleted and the Julia version of the same notebook moved to `track_spin`. The Python example therefore fails at its first spin step. Users of the Python spin-ramp example are affected; the failure is loud.

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
    # therefore fails at its first spin step. Users of the Python spin-ramp example are affected; the
    # failure is loud.
    #
    # The script prints the notebook lines that call `jl.rotate_spins` and checks whether SciBmad
    # defines `rotate_spins`, `rotate_spins!`, `track_spin` and `track_spin!`; the example's call target
    # should exist. Set `SCIBMAD_SOURCE` to a SciBmad.jl checkout, from which the notebook is read.
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
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |

