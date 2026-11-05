---
title: "LAT-20: Bmad SciBmad writer emits reverse(ele) without a warning; no such method exists"
hide:
  - toc
---

# LAT-20 — Bmad SciBmad writer emits `reverse(ele)` without a warning; no such method exists

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on bmad-ecosystem 20261020-0 (`19b8afb`) |
| Tags | <span class="bb-tag">known-limitation</span> [?](../../methodology.md#tags) |
| Source | [`bmad/bmad/output/write_lattice_scibmad_format.f90:875-879`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L875-L879)<br>[`BeamTracking.jl/ext/BeamTrackingBeamlinesExt/general_bl.jl:13`](https://github.com/bmad-sim/BeamTracking.jl/blob/d8e45fdc25dbebf4be8195e69be04df13d7383b1/ext/BeamTrackingBeamlinesExt/general_bl.jl#L13)<br>[`bmad/util_programs/bmad_to_scibmad/bmad_to_scibmad.f90:56-58`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/util_programs/bmad_to_scibmad/bmad_to_scibmad.f90#L56-L58) |

For a Bmad line containing reversed elements, such as `line = (--d)`, the exporter `bmad_to_scibmad` writes `reverse(ele)` calls, and loading the file fails with `MethodError: no method matching reverse(::LineElement)`. Beamlines defines no element-orientation support, and BeamTracking's source explicitly marks reversed elements as future work; a runtime method check independently confirms that `reverse(::LineElement)` does not exist. The exporter gives no incomplete-translation warning for the reproduced lattice.

??? example "Reproducer: LAT-20.jl"

    [Download LAT-20.jl](LAT-20.jl)

    ```julia
    # LAT-20: Bmad SciBmad writer emits `reverse(ele)` without a warning; no such method exists
    #
    # For a Bmad line containing reversed elements, such as `line = (--d)`, the exporter
    # `bmad_to_scibmad` writes `reverse(ele)` calls, and loading the file fails with `MethodError: no
    # method matching reverse(::LineElement)`. Beamlines defines no element-orientation support, and
    # BeamTracking's source explicitly marks reversed elements as future work; a runtime method check
    # independently confirms that `reverse(::LineElement)` does not exist. The exporter gives no
    # incomplete-translation warning for the reproduced lattice.
    #
    # The script exports `lat: line = (--d)`, with `d` a 1 m drift, using Bmad's `bmad_to_scibmad`
    # program, reports whether the translator flagged an incomplete translation, and loads the emitted
    # file; a loadable lattice or a translator warning is expected. It also checks whether a
    # `reverse(::LineElement)` method exists. Set `BMAD_SOURCE` to the root of a Bmad build, so that the
    # program is `$BMAD_SOURCE/production/bin/bmad_to_scibmad`.
    using SciBmad
    haskey(ENV, "BMAD_SOURCE") || error("set BMAD_SOURCE to the root of the bmad-ecosystem 20261020-0 build (the directory that contains production/bin)")
    const BIN = joinpath(ENV["BMAD_SOURCE"], "production", "bin", "bmad_to_scibmad")
    isfile(BIN) || error("bmad_to_scibmad not found at $BIN")
    dir = mktempdir()
    b = joinpath(dir, "rev.bmad"); j = joinpath(dir, "rev.jl")
    write(b, """
    parameter[particle] = proton
    parameter[p0c] = 1e9
    parameter[geometry] = open
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    d: drift, l = 1
    lat: line = (--d)
    use, lat
    """)
    log = read(pipeline(Cmd(`$BIN $b $j`; dir=dir, env=merge(ENV, Dict("OMP_NUM_THREADS" => "1"))), stderr=stdout), String)
    println("translator reported incomplete translation: ", occursin("NOT COMPLETE", log))
    src = read(j, String)
    println("emitted: ", strip(only(filter(l -> occursin("Branch(", l), split(src, '\n')))))
    mod = Module(); Core.eval(mod, :(using SciBmad))
    r = try
        Base.include_string(mod, replace(src, r"^using .*$"m => "")); "loaded"
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("loading emitted file: ", r)
    # Beamlines has no element-orientation support; BeamTracking fixes the element orientation to 1.
    println("hasmethod(reverse, Tuple{LineElement}) = ", hasmethod(reverse, Tuple{LineElement}))
    println("expected: a loadable lattice, or a translator error/warning for the unsupported orientation")
    println(startswith(r, "ERROR") ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU; emitted constructs typed by hand, the exporter was not built |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |

