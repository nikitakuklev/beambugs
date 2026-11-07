# LAT-22: Bmad SciBmad writer's Context `c1` overwrites a Bmad element named `c1`
#
# Bmad's SciBmad exporter binds the lattice's variable Context to the name `c1` after defining
# elements, overwriting any element whose name becomes `c1`. Bmad names are case-insensitive, so
# both `c1` and `C1` trigger the collision. Loading then fails with “Unable to construct Branch:
# entry 2 of elements is a Context{Any}”, while the same lattice with element name `c2` loads. A
# Branch named `c1` instead overwrites the Context and also prevents loading; renaming only the
# generated Context binding and its final reference makes the tested collisions load. The exporter
# gives no incomplete-translation warning.
#
# Setup: `BMAD_SOURCE` set to the root of a bmad-ecosystem `20261020-0` build (the directory
# containing `production/bin`); the script runs its `bmad_to_scibmad` program.
#
# The script exports a one-quadrupole lattice with the quadrupole named `c1`, `C1` and `c2` in turn
# using Bmad's `bmad_to_scibmad` program and loads each file; every name should load, and `c2` is
# the control. After each load it prints what the name `c1` holds.
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
