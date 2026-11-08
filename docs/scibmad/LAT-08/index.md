---
title: "LAT-08: track rejects the documented vector weight"
hide:
  - toc
---

# LAT-08 — `track` rejects the documented vector `weight`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-31 on SciBmad commit [`ae9423d`](https://github.com/bmad-sim/SciBmad.jl/commit/ae9423d703b4), committed 2026-10-28 |
| Source | [`SciBmad.jl/src/track.jl:182`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L182)<br>[`SciBmad.jl/src/track.jl:130-131`](https://github.com/bmad-sim/SciBmad.jl/blob/ae9423d703b4e612ae66e9deb0546aa79f62a340/src/track.jl#L130-L131)<br>[`BeamTracking.jl/src/types.jl:78-79`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/types.jl#L78-L79)<br>[`BeamTracking.jl/src/types.jl:102-114`](https://github.com/bmad-sim/BeamTracking.jl/blob/0bf3585fdc9c5b41859b23b92854376b933a4fe6/src/types.jl#L102-L114) |

Passing per-particle weights to `track` as the documented vector of length `n_particles` raises a `TypeError` before tracking begins, because the keyword is declared `Union{AbstractMatrix,Nothing}`. `track(...; weight=[0.25, 0.75])` therefore fails on documented input. The same vector passed through an explicit `Bunch` is kept as [0.25, 0.75] in a zero-turn control, confirming that the underlying bunch API accepts it.

Run `julia LAT-08.jl`.

??? example "Reproducer: LAT-08.jl"

    [Download LAT-08.jl](LAT-08.jl)

    ```julia
    # LAT-08: `track` rejects the documented vector `weight`
    #
    # Passing per-particle weights to `track` as the documented vector of length `n_particles` raises a
    # `TypeError` before tracking begins, because the keyword is declared
    # `Union{AbstractMatrix,Nothing}`. `track(...; weight=[0.25, 0.75])` therefore fails on documented
    # input. The same vector passed through an explicit `Bunch` is kept as [0.25, 0.75] in a zero-turn
    # control, confirming that the underlying bunch API accepts it.
    #
    # The script tracks two electrons through a 1 m drift for zero turns, passing `weight=[0.25, 0.75]`
    # to `track`, and as a control passes the same weights inside an explicit `Bunch`. The resulting
    # bunch must carry the weights [0.25, 0.75].
    using SciBmad
    bl = Beamline([Drift(L=1.0)], species_ref=Species("electron"), E_ref=1e9)
    v0 = [1e-3 1e-4 0 0 0 0; 2e-3 0 0 0 0 0]
    w = [0.25, 0.75]
    r = try
        res = track(bl; v0=v0, weight=w, n_turns=0)
        res.bunch.weight
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("observed track(...; weight=[0.25,0.75]) -> bunch.weight = ", r)
    b = Bunch(v=copy(v0), weight=copy(w), species=bl.species_ref, p_over_q_ref=bl.p_over_q_ref)
    res2 = track(bl; bunch=b, n_turns=0)
    println("control  track(...; bunch=Bunch(..., weight=[0.25,0.75])) -> bunch.weight = ", res2.bunch.weight)
    println("expected bunch.weight = ", w)
    println(r == w ? "NOT REPRODUCED" : "DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | <span class="bb-nowrap">`gpt-6-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-22</span> | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-10-22</span> | SciBmad 0.6.0 (`62399f930847`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

