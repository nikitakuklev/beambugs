---
title: "LAT-21: Bmad SciBmad writer drops the last exponent digit of aperture limits"
hide:
  - toc
---

# LAT-21 — Bmad SciBmad writer drops the last exponent digit of aperture limits

| | |
|---|---|
| Code | [SciBmad](../index.md) · Lattice and API |
| Priority | P1 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-22 on bmad-ecosystem 20261020-0 (`19b8afb`, committed 2026-10-20) |
| Source | [`bmad/bmad/output/write_lattice_scibmad_format.f90:685-688`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L685-L688)<br>[`:845-858`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_scibmad_format.f90#L845-L858)<br>[`bmad/bmad/output/write_lattice_file_mod.f90:184-199`](https://github.com/bmad-sim/bmad-ecosystem/blob/19b8afb46a400d12a53a92d616e02359b3b2b71f/bmad/output/write_lattice_file_mod.f90#L184-L199)<br>[`Beamlines.jl/src/aperture.jl:17-20`](https://github.com/bmad-sim/Beamlines.jl/blob/35fe0162c72d0899a02a67eb929e8a900a165bc1/src/aperture.jl#L17-L20) |

Bmad's SciBmad exporter writes aperture limits through a 24-character string, dropping the final exponent digit of the 25-character number format. A nonzero limit a·10ᵉ m therefore loads as a·10ᵗ m, with t = e/10 truncated toward zero; only magnitudes in [1, 10) m keep their exponent. Half-apertures of 0.01 m, 0.3 m and 15 m load as 1 m, 3 m and 1.5 m, and a particle at x = 0.02 m passes a collimator whose intended half-aperture is 0.01 m. An independent model of the format reproduces the loaded values, and tracking confirms the missed loss. The wrong limits load without a warning.

Run `julia LAT-21.jl`.

??? example "Reproducer: LAT-21.jl"

    [Download LAT-21.jl](LAT-21.jl)

    ```julia
    # LAT-21: Bmad SciBmad writer drops the last exponent digit of aperture limits
    #
    # Bmad's SciBmad exporter writes aperture limits through a 24-character string, dropping the final
    # exponent digit of the 25-character number format. A nonzero limit a·10ᵉ m therefore loads as a·10ᵗ
    # m, with t = e/10 truncated toward zero; only magnitudes in [1, 10) m keep their exponent.
    # Half-apertures of 0.01 m, 0.3 m and 15 m load as 1 m, 3 m and 1.5 m, and a particle at x = 0.02 m
    # passes a collimator whose intended half-aperture is 0.01 m. An independent model of the format
    # reproduces the loaded values, and tracking confirms the missed loss. The wrong limits load without
    # a warning.
    #
    # Setup: `BMAD_SOURCE` set to the root of a bmad-ecosystem `20261020-0` build (the directory
    # containing `production/bin`); the script runs its `bmad_to_scibmad` program.
    #
    # The script exports three collimators with half-apertures (x, y) = (0.01, 0.3), (2, 15) and
    # (0.00025, 120) m using Bmad's `bmad_to_scibmad` program, loads the file and compares each limit
    # with Bmad's value and with a Printf model of the truncated Fortran format. It then tracks a
    # particle at x = 0.02 m through the collimators, which must lose it at the 0.01 m limit.
    using SciBmad, Printf
    const BT = SciBmad.BeamTracking
    haskey(ENV, "BMAD_SOURCE") || error("set BMAD_SOURCE to the root of the bmad-ecosystem 20261020-0 build (the directory that contains production/bin)")
    const BIN = joinpath(ENV["BMAD_SOURCE"], "production", "bin", "bmad_to_scibmad")
    isfile(BIN) || error("bmad_to_scibmad not found at $BIN")
    dir = mktempdir()
    b = joinpath(dir, "ap.bmad"); j = joinpath(dir, "ap.jl")
    write(b, """
    parameter[particle] = proton
    parameter[p0c] = 1e9
    parameter[geometry] = open
    beginning[beta_a] = 10
    beginning[beta_b] = 10
    ca: rcollimator, l = 0.1, x_limit = 0.01, y_limit = 0.3
    cb: rcollimator, l = 0.1, x_limit = 2.0, y_limit = 15.0
    cc: ecollimator, l = 0.1, x_limit = 2.5e-4, y_limit = 120
    lat: line = (ca, cb, cc)
    use, lat
    """)
    log = read(pipeline(Cmd(`$BIN $b $j`; dir=dir, env=merge(ENV, Dict("OMP_NUM_THREADS" => "1"))), stderr=stdout), String)
    src = read(j, String)
    println("translator reported incomplete translation: ", occursin("NOT COMPLETE", log))
    println("emitted: ", join([m.match for m in eachmatch(r"(L|[xy]2_limit)\s*=\s*\S+", src)], " | "))
    mod = Module(); Core.eval(mod, :(using SciBmad))
    Base.include_string(mod, replace(src, r"^using .*$"m => ""))
    bl = Core.eval(mod, :lat).branches[1].beamlines[1]
    el = Dict(e.name => e for e in bl.line)
    bmad = [("ca", :x2_limit, 0.01), ("ca", :y2_limit, 0.3), ("cb", :x2_limit, 2.0), ("cb", :y2_limit, 15.0),
            ("cc", :x2_limit, 2.5e-4), ("cc", :y2_limit, 120.0)]
    # re_str writes es25.17e3 into 25 characters: sign, 19 mantissa characters, "E", the exponent sign and three
    # exponent digits. aper_str returns character(24), which cuts the last exponent digit, so a*10^e is emitted
    # as a*10^trunc(e/10). Other attributes go through re_str into the long line and keep all digits (L below).
    # es25.17e3: Fortran prints d.ddd...E+eee; truncate to 24 characters.
    fortran24(v) = (s = @sprintf("%25.17E", v); m = match(r"E([+-])(\d+)$", s); s = s[1:end - length(m[2]) - 1] * m[1] * lpad(m[2], 3, '0');
                    s = lpad(s, 25); s[1:24])
    nbad = 0
    for (n, k, v) in bmad
        got = getproperty(el[n], k); model = parse(Float64, fortran24(v))
        global nbad += got != v
        println(rpad("$n.$k", 14), " Bmad = ", rpad(v, 8), " loaded = ", rpad(got, 22), " independent: Printf model of the truncation = ", model, " (\"", strip(fortran24(v)), "\")")
    end
    println("loaded L of ca = ", el["ca"].L, " (re_str into the long line keeps the exponent)")
    r = track(bl; v0=[0.02 0 0 0 0 0])
    println("independent: particle at x = 0.02 m, final state = ", r.state[1, end], " (expected STATE_LOST_POS_X = ", BT.STATE_LOST_POS_X, " at the 0.01 m limit)")
    println(nbad > 0 && r.state[1, end] != BT.STATE_LOST_POS_X ? "DEFECT REPRODUCED" : "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `claude-opus-5-5` | high | 2026-10-22 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, SciBmad 0.6.0 release packages, Julia 1.13.2, CPU |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | bmad-ecosystem 20261020-0 (`19b8afb46a40`), production bmad_to_scibmad 20261020-0, source (SciBmad 5f0fb83ac889, BeamTracking d8e45fdc25db, Beamlines 35fe0162c72d), Julia 1.13.2, CPU, one thread |

