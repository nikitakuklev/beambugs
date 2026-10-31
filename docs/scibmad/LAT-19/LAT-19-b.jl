# LAT-19: four more emitted names that Beamlines rejects: `roll`, `z_rot`, `t_offset` and `is_on`
#
# The script exports four Bmad lattices with `bmad_to_scibmad`, one per name: a rolled bend
# (`roll`), an overlay on a quadrupole's tilt (`z_rot`), an overlay on a patch's `t_offset`, and a
# switched-off quadrupole (`is_on`). Each emitted file should load. As independent checks it
# confirms that the emitted name is not a Beamlines property and that the file with the Beamlines
# name substituted (`tilt` or `dt`, or `is_on` dropped) loads with Bmad's value; it also prints the
# emitted line of a switched-off bend, which keeps its field. Set `BMAD_SOURCE` to the root of a
# Bmad build, so that the program is `$BMAD_SOURCE/production/bin/bmad_to_scibmad`.
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
