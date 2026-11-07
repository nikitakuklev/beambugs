---
title: "LAT-14: TrackingResult.elapsed_time documented in minutes, stores seconds"
hide:
  - toc
---

# LAT-14 — `TrackingResult.elapsed_time` documented in minutes, stores seconds

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-31 on SciBmad commit [`ae9423d`](https://github.com/bmad-sim/SciBmad.jl/commit/ae9423d703b4), committed 2026-10-28 |
| Source | [`SciBmad.jl/src/track.jl:76`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L76)<br>[`:84`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L84)<br>[`:277`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L277)<br>[`:101`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L101) |

The `TrackingResult` docstring and field comment state that `elapsed_time` is in minutes, but the value comes from Julia's `@elapsed`, which returns seconds, and the result's own display labels it seconds. For 10⁵ particles tracked over 10 000 turns, `elapsed_time` is 0.9718 while a wall clock around `track` reads 0.9767 s, a ratio of 0.9949, where a value in minutes would give 1/60 = 0.01667. Tracking results are unaffected.

Run `julia LAT-14.jl`.

??? example "Reproducer: LAT-14.jl"

    [Download LAT-14.jl](LAT-14.jl)

    ```julia
    # LAT-14: `TrackingResult.elapsed_time` documented in minutes, stores seconds
    #
    # The `TrackingResult` docstring and field comment state that `elapsed_time` is in minutes, but the
    # value comes from Julia's `@elapsed`, which returns seconds, and the result's own display labels it
    # seconds. For 10⁵ particles tracked over 10 000 turns, `elapsed_time` is 0.9718 while a wall clock
    # around `track` reads 0.9767 s, a ratio of 0.9949, where a value in minutes would give 1/60 =
    # 0.01667. Tracking results are unaffected.
    #
    # The script tracks 10⁵ electrons through a 1 m drift for 10 000 turns, times the `track` call with
    # a wall clock and divides `elapsed_time` by that time. The ratio is near 1 for seconds and 1/60 for
    # minutes; the defect is reported when it lies in (0.5, 1].
    using SciBmad
    bl = Beamline([Drift(L=1.0)], species_ref=Species("electron"), E_ref=1e9)
    v0 = zeros(100_000, 6); v0[:, 1] .= 1e-3; v0[:, 2] .= 1e-4
    track(bl; v0=v0, n_turns=1)                       # compile
    n = 10_000
    t0 = time_ns()
    res = track(bl; v0=v0, n_turns=n, save_every_n_turns=n)
    wall = (time_ns() - t0) / 1e9
    println("wall-clock seconds around track() = ", wall)
    println("res.elapsed_time                  = ", res.elapsed_time)
    println("ratio elapsed_time / wall_seconds = ", res.elapsed_time / wall, " (1 if seconds, 1/60 = ", 1 / 60, " if minutes)")
    println("docstring: \"The tracking execution time in minutes\"")
    r = res.elapsed_time / wall
    println(r > 0.5 && r <= 1.0 && wall > 0.2 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Found | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

