---
title: "LAT-13: Bunch docstring gives a three-argument callback; tracking calls nine"
hide:
  - toc
---

# LAT-13 — `Bunch` docstring gives a three-argument callback; tracking calls nine

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-22 on BeamTracking 0.8.7 (`65e343f`); not reproduced on BeamTracking main (`0bf3585`), 2026-10-29 |
| Pull request | [[gh:https://github.com/bmad-sim/BeamTracking.jl/pull/370]], fix |
| Source | [`BeamTracking.jl/src/types.jl:80-86`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/types.jl#L80-L86)<br>[`BeamTracking.jl/src/callbacks.jl:61`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/src/callbacks.jl#L61)<br>[`SciBmad.jl/docs/src/track.md:155-177`](https://github.com/bmad-sim/SciBmad.jl/blob/5f0fb83ac8899c1e26d4522ccc8e54c59633307f/docs/src/track.md#L155-L177) |

The `Bunch` docstring tells users to write a tracking callback as `callback(coords, ds_step, g)`, but tracking calls each callback with nine arguments, `(i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!)`. The call site and the narrative tracking documentation agree on the nine-argument form. A callback written to the docstring raises a `MethodError` at its first call, while a nine-argument callback runs. The failure is loud.

??? example "Reproducer: LAT-13.jl"

    [Download LAT-13.jl](LAT-13.jl)

    ```julia
    # LAT-13: `Bunch` docstring gives a three-argument callback; tracking calls nine
    #
    # The `Bunch` docstring tells users to write a tracking callback as `callback(coords, ds_step, g)`,
    # but tracking calls each callback with nine arguments, `(i, coords, cur_s, cur_t_ref,
    # cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!)`. The call site and the
    # narrative tracking documentation agree on the nine-argument form. A callback written to the
    # docstring raises a `MethodError` at its first call, while a nine-argument callback runs. The
    # failure is loud.
    #
    # The script reads the `Bunch` docstring of the loaded package and tracks an electron through a 1 m
    # drift once with a callback of the three-argument form and once with a nine-argument callback. It
    # reports the defect only while the docstring prescribes the three-argument form, that callback
    # fails and the nine-argument one runs, so its verdict reflects the build it runs on.
    using SciBmad
    const THREE = "callback(coords::Coords, ds_step, g)"
    const NINE = "callback(i, coords, cur_s, cur_t_ref, cur_beta_gamma_ref, last_ds_step, last_g, transforms_out!, transforms_in!)"
    doc = replace(string(@doc SciBmad.BeamTracking.Bunch), r"\s+" => " ")
    doc_three = occursin(THREE, doc)
    doc_nine = occursin(NINE, doc)
    println("Bunch docstring of the loaded BeamTracking: prescribes three-argument form = ", doc_three, ", nine-argument form = ", doc_nine)
    bl = Beamline([Drift(L=1.0)], species_ref=Species("electron"), E_ref=1e9)
    v0 = [1e-3 0 0 0 0 0]
    doc_cb = (coords, ds_step, g) -> nothing
    r3 = try
        track(bl; v0=v0, callbacks=(doc_cb,)); "ran"
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("docstring-form callback (coords, ds_step, g): ", r3)
    n = Ref(0)
    cb9 = (i, coords, s, t, bg, ds, g, tout!, tin!) -> (n[] += 1; nothing)
    track(bl; v0=v0, callbacks=(cb9,))
    println("nine-argument callback: ran, called ", n[], " time(s)")
    println("expected (per docstring): the callback of the documented form runs")
    println(doc_three && !doc_nine && startswith(r3, "ERROR") && n[] > 0 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6-sol` | xhigh | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | BeamTracking 0.8.7 (`65e343f9afda`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

