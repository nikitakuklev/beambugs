---
title: "LAT-19: Bmad SciBmad writer emits property names Beamlines does not define (bsol_field, roll, is_on, controlled z_rot and t_offset)"
hide:
  - toc
---

# LAT-19 — Bmad SciBmad writer emits property names Beamlines does not define (`bsol_field`, `roll`, `is_on`, controlled `z_rot` and `t_offset`)

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on bmad-ecosystem 20261020-0 (`19b8afb`, committed 2026-10-20) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`bmad/bmad/output/write_lattice_scibmad_format.f90:777`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L777)<br>[`:633`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L633)<br>[`:622`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L622)<br>[`:1233`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L1233)<br>[`:1237`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L1237)<br>[`Beamlines.jl/src/keymaps.jl:2`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/keymaps.jl#L2)<br>[`Beamlines.jl/src/element.jl:481`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/element.jl#L481)<br>[`Beamlines.jl/src/bmultipole.jl:240`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/bmultipole.jl#L240)<br>[`Beamlines.jl/src/keymaps.jl:73`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/keymaps.jl#L73)<br>[`Beamlines.jl/src/alignment.jl:3-21`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/alignment.jl#L3-L21)<br>[`Beamlines.jl/src/keymaps.jl:75`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/keymaps.jl#L75)<br>[`bmad/bmad/output/write_lattice_scibmad_format.f90:1239`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L1239)<br>[`:769`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L769)<br>[`:749`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L749) |

Bmad's SciBmad exporter emits property names Beamlines rejects: `bsol_field` for field-master solenoids, `roll` for rolled bends, `is_on` for switched-off elements, and `z_rot` or `t_offset` in controlled assignments. Executed solenoid, bend, switched-off quadrupole and overlay cases fail at load with “Type LineElement has no property” and no incomplete-translation warning. The accepted names are `Bsol`, `tilt` for a bend roll or non-patch tilt, and `dt` for a patch time shift; substituting them loads the tested cases with their expected values. Beamlines has no on/off switch, and dropping `is_on` leaves an off bend's nonzero exported field. Source inspection extends the controlled-name failures to groups and identifies a warning for a controlled roll.

Run `julia LAT-19.jl`.

??? example "Reproducer: LAT-19.jl"

    [Download LAT-19.jl](LAT-19.jl)

    ```julia
    # LAT-19: Bmad SciBmad writer emits property names Beamlines does not define (`bsol_field`, `roll`,
    # `is_on`, controlled `z_rot` and `t_offset`)
    #
    # Bmad's SciBmad exporter emits property names Beamlines rejects: `bsol_field` for field-master
    # solenoids, `roll` for rolled bends, `is_on` for switched-off elements, and `z_rot` or `t_offset`
    # in controlled assignments. Executed solenoid, bend, switched-off quadrupole and overlay cases fail
    # at load with “Type LineElement has no property” and no incomplete-translation warning. The
    # accepted names are `Bsol`, `tilt` for a bend roll or non-patch tilt, and `dt` for a patch time
    # shift; substituting them loads the tested cases with their expected values. Beamlines has no
    # on/off switch, and dropping `is_on` leaves an off bend's nonzero exported field. Source inspection
    # extends the controlled-name failures to groups and identifies a warning for a controlled roll.
    #
    # Setup: `BMAD_SOURCE` set to the root of a bmad-ecosystem `20261020-0` build (the directory
    # containing `production/bin`); the script runs its `bmad_to_scibmad` program.
    #
    # The script exports a field-master solenoid (`l = 1`, `bs_field = 0.5`) with Bmad's
    # `bmad_to_scibmad` program, prints the emitted solenoid line and loads the file, which should give
    # a 0.5 T solenoid field; a directly built `Solenoid(L=1, Bsol=0.5)` is the control. The other names
    # are covered by a second script.
    using SciBmad
    haskey(ENV, "BMAD_SOURCE") || error("set BMAD_SOURCE to the root of the bmad-ecosystem 20261020-0 build (the directory that contains production/bin)")
    const BIN = joinpath(ENV["BMAD_SOURCE"], "production", "bin", "bmad_to_scibmad")
    isfile(BIN) || error("bmad_to_scibmad not found at $BIN")
    dir = mktempdir()
    b = joinpath(dir, "sol.bmad"); j = joinpath(dir, "sol.jl")
    write(b, """
    parameter[particle] = proton
    parameter[p0c] = 1e9
    parameter[geometry] = open
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    sol: solenoid, l = 1, bs_field = 0.5, field_master = T
    lat: line = (sol)
    use, lat
    """)
    run(pipeline(Cmd(`$BIN $b $j`; dir=dir, env=merge(ENV, Dict("OMP_NUM_THREADS" => "1"))), stdout=devnull, stderr=devnull))
    src = read(j, String)
    println("emitted: ", strip(only(filter(l -> occursin("Solenoid(", l), split(src, '\n')))))
    mod = Module(); Core.eval(mod, :(using SciBmad))
    r = try
        Base.include_string(mod, replace(src, r"^using .*$"m => "")); "loaded"
    catch e
        "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
    println("loading emitted file: ", r)
    # Beamlines names the solenoid field `Bsol`; the writer emits `bsol_field` for field_master solenoids.
    s = Solenoid(L=1.0, Bsol=0.5)
    println("control Solenoid(L=1, Bsol=0.5).Bsol = ", s.Bsol)
    println("expected: emitted file loads with a 0.5 T solenoid field")
    println(startswith(r, "ERROR") ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

??? example "Reproducer: LAT-19-b.jl"

    [Download LAT-19-b.jl](LAT-19-b.jl)

    ```julia
    # LAT-19: four more emitted names that Beamlines rejects: `roll`, `z_rot`, `t_offset` and `is_on`
    #
    # Setup: `BMAD_SOURCE` set to the root of a bmad-ecosystem `20261020-0` build (the directory
    # containing `production/bin`); the script runs its `bmad_to_scibmad` program.
    #
    # The script exports four Bmad lattices with `bmad_to_scibmad`, one per name: a rolled bend
    # (`roll`), an overlay on a quadrupole's tilt (`z_rot`), an overlay on a patch's `t_offset`, and a
    # switched-off quadrupole (`is_on`). Each emitted file should load. As independent checks it
    # confirms that the emitted name is not a Beamlines property and that the file with the Beamlines
    # name substituted (`tilt` or `dt`, or `is_on` dropped) loads with Bmad's value; it also prints the
    # emitted line of a switched-off bend, which keeps its field.
    using SciBmad
    const BL = SciBmad.Beamlines
    haskey(ENV, "BMAD_SOURCE") || error("set BMAD_SOURCE to the root of the bmad-ecosystem 20261020-0 build (the directory that contains production/bin)")
    const BIN = joinpath(ENV["BMAD_SOURCE"], "production", "bin", "bmad_to_scibmad")
    isfile(BIN) || error("bmad_to_scibmad not found at $BIN")
    HDR = "parameter[particle] = electron\nparameter[p0c] = 1e9\nparameter[geometry] = open\nbeginning[beta_a] = 10\nbeginning[beta_b] = 10\n"
    function export_src(body)
        dir = mktempdir(); b = joinpath(dir, "x.bmad"); j = joinpath(dir, "x.jl")
        write(b, HDR * body)
        log = read(pipeline(Cmd(`$BIN $b $j`; dir=dir, env=merge(ENV, Dict("OMP_NUM_THREADS" => "1"))), stderr=stdout), String)
        read(j, String), occursin("NOT COMPLETE", log)
    end
    function load(src)
        mod = Module(); Core.eval(mod, :(using SciBmad))
        try
            Base.include_string(mod, replace(src, r"^using .*$"m => "")); (Core.eval(mod, :lat), "loaded")
        catch e
            (nothing, "ERROR: " * first(split(sprint(showerror, e), '\n')))
        end
    end
    # Beamlines rejects names that are not keys of PROPERTIES_MAP. Its names for these are AlignmentParams `tilt`
    # (bend roll, controlled tilt of a non-patch) and PatchParams `dt` (patch time shift, which the uncontrolled
    # patch line already writes); Beamlines has no on/off switch.
    cases = [("roll", "b: sbend, l = 2, angle = 0.1, roll = 0.01\nln: line = (b)\nuse, ln\n", r"\broll =", "tilt =", "b", :tilt, 0.01),
             ("z_rot", "q: quadrupole, l = 0.5, k1 = 0.3\nov: overlay = {q[tilt]: tv}, var = {tv}, tv = 0.02\nln: line = (q)\nuse, ln\n", r"\.z_rot", ".tilt", "q", :tilt, 0.02),
             ("t_offset", "p: patch, t_offset = 1e-9\nov: overlay = {p[t_offset]: tv}, var = {tv}, tv = 3e-9\nln: line = (p)\nuse, ln\n", r"\.t_offset", ".dt", "p", :dt, 3e-9),
             ("is_on", "q: quadrupole, l = 0.5, k1 = 0.3, is_on = F\nln: line = (q)\nuse, ln\n", r"is_on = false,\s*", "", "q", :Kn1, 0.0)]
    nbad = 0
    for (key, body, pat, rep, ename, prop, want) in cases
        src, inc = export_src(body)
        _, r = load(src)
        global nbad += startswith(r, "ERROR")
        lat2, r2 = load(replace(src, pat => rep))
        got = isnothing(lat2) ? r2 : getproperty(only(filter(e -> e.name == ename, lat2.branches[1].beamlines[1].line)), prop)
        println(rpad(key, 9), " emitted: ", occursin(Regex(key), src), "  incomplete warning: ", inc, "  load -> ", r)
        println(rpad("", 9), " independent: haskey(PROPERTIES_MAP, :$key) = ", haskey(BL.PROPERTIES_MAP, Symbol(key)),
                isempty(rep) ? "" : ", :$(prop) defined = $(haskey(BL.PROPERTIES_MAP, prop))",
                "; with '", isempty(rep) ? "is_on dropped" : rep, "' the file loads and $ename.$prop = ", got, " (Bmad value ", want, ")")
    end
    # is_on = F needs per-type handling: an off bend still gets its Kn0 from the angle (writer :658-661).
    src, _ = export_src("b: sbend, l = 2, angle = 0.1, is_on = F\nln: line = (b)\nuse, ln\n")
    println("off sbend emitted: ", strip(join(filter(l -> occursin("SBend", l) || occursin("Kn0", l), split(src, '\n')), " ")))
    println(nbad == length(cases) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), SciBmad 0.6.0 release packages, Julia 1.13.2, CPU; exporter output written by hand, the exporter was not run |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed<br><small>roll, is_on, z_rot and t_offset names</small> | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |

