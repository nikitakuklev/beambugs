---
title: "LAT-22: Bmad SciBmad writer's Context c1 overwrites a Bmad element named c1"
hide:
  - toc
---

# LAT-22 — Bmad SciBmad writer's Context `c1` overwrites a Bmad element named `c1`

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-30 on bmad-ecosystem 20261020-0 (`19b8afb`) |
| Source | [`bmad/bmad/output/write_lattice_scibmad_format.f90:350`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L350)<br>[`:480`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L480)<br>[`:888-910`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L888-L910) |

Bmad's SciBmad exporter binds the lattice's variable Context to the name `c1` after defining elements, overwriting any element whose name becomes `c1`. Bmad names are case-insensitive, so both `c1` and `C1` trigger the collision. Loading then fails with “Unable to construct Branch: entry 2 of elements is a Context{Any}”, while the same lattice with element name `c2` loads. A Branch named `c1` instead overwrites the Context and also prevents loading; renaming only the generated Context binding and its final reference makes the tested collisions load. The exporter gives no incomplete-translation warning, so affected Bmad users discover the problem at load.

??? example "Reproducer: LAT-22.jl"

    [Download LAT-22.jl](LAT-22.jl)

    ```julia
    # LAT-22: Bmad SciBmad writer's Context `c1` overwrites a Bmad element named `c1`
    #
    # Bmad's SciBmad exporter binds the lattice's variable Context to the name `c1` after defining
    # elements, overwriting any element whose name becomes `c1`. Bmad names are case-insensitive, so
    # both `c1` and `C1` trigger the collision. Loading then fails with “Unable to construct Branch:
    # entry 2 of elements is a Context{Any}”, while the same lattice with element name `c2` loads. A
    # Branch named `c1` instead overwrites the Context and also prevents loading; renaming only the
    # generated Context binding and its final reference makes the tested collisions load. The exporter
    # gives no incomplete-translation warning, so affected Bmad users discover the problem at load.
    #
    # The script exports a one-quadrupole lattice with the quadrupole named `c1`, `C1` and `c2` in turn
    # using Bmad's `bmad_to_scibmad` program and loads each file; every name should load, and `c2` is
    # the control. After each load it prints what the name `c1` holds. Set `BMAD_SOURCE` to the root of
    # a Bmad build, so that the program is `$BMAD_SOURCE/production/bin/bmad_to_scibmad`.
    using SciBmad
    haskey(ENV, "BMAD_SOURCE") || error("set BMAD_SOURCE to the root of the bmad-ecosystem 20261020-0 build (the directory that contains production/bin)")
    const BIN = joinpath(ENV["BMAD_SOURCE"], "production", "bin", "bmad_to_scibmad")
    isfile(BIN) || error("bmad_to_scibmad not found at $BIN")
    # The writer emits `c1 = Context(` after the element definitions and passes `context = c1` to the lattice.
    # Element names are downcased and otherwise emitted verbatim as Julia bindings, so `c1` and `C1` collide.
    function export_load(name)
        dir = mktempdir(); b = joinpath(dir, "x.bmad"); j = joinpath(dir, "x.jl")
        write(b, """
        parameter[particle] = electron
        parameter[p0c] = 1e9
        parameter[geometry] = open
        beginning[beta_a] = 10
        beginning[beta_b] = 10
        $name: quadrupole, l = 0.5, k1 = 0.3
        ln: line = ($name)
        use, ln
        """)
        log = read(pipeline(Cmd(`$BIN $b $j`; dir=dir, env=merge(ENV, Dict("OMP_NUM_THREADS" => "1"))), stderr=stdout), String)
        src = read(j, String)
        mod = Module(); Core.eval(mod, :(using SciBmad))
        r = try
            Base.include_string(mod, replace(src, r"^using .*$"m => "")); "loaded"
        catch e
            "ERROR: " * first(split(sprint(showerror, e), '\n'))
        end
        c1 = try Core.eval(mod, :(typeof(c1))) catch; "undefined" end
        (r, occursin("NOT COMPLETE", log), filter(l -> occursin(r"c1 = |Branch\(", l), split(src, '\n')), c1)
    end
    for name in ("c1", "C1", "c2")
        r, inc, lines, t = export_load(name)
        println("element $name: incomplete warning = $inc; emitted ", join(strip.(lines), " ... "), "; load -> ", r, "; typeof(c1) = ", t)
    end
    r1 = export_load("c1")[1]; r2 = export_load("c2")[1]
    println("expected: every name loads (control c2: ", r2, ")")
    println(startswith(r1, "ERROR") && r2 == "loaded" ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, Julia 1.13.2, installed SciBmad 0.6.0 package set (SciBmad 62399f930847, BeamTracking 65e343f9afda, Beamlines 35fe0162c72d), CPU |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, Julia 1.13.2, pinned SciBmad package set (SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), CPU, one thread |

