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
