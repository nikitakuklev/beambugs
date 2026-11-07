# LAT-18: Bmad SciBmad writer copies `phi0` (rad/2π) into a radians field with the wrong sign
# convention
#
# Bmad's SciBmad exporter copies an rfcavity's `phi0`, which Bmad gives in turns (rad/2π), into
# SciBmad's radians field and selects `PhaseRef.AboveTransition`, introducing both a scale and a
# sign error. The required SciBmad phase is φ₀ = −2π·(`phi0` + `phi0_err`), and the export also
# omits `phi0_err`. For a 1 m, 1 MV, 100 MHz cavity and a 1 GeV/c proton, `phi0 = 0.25` gives a
# relative momentum deviation of −3.393×10⁻⁴ instead of 1.371×10⁻³ from Bmad's program Tao, while
# `phi0 = 1` gives −1.154×10⁻³ instead of zero. Hand-built cavities using the corrected mapping
# agree with Tao within a relative 2.500×10⁻⁴. Source inspection shows the 2π scaling missing for
# lcavity export too.
#
# Setup: `BMAD_SOURCE` set to the root of a bmad-ecosystem `20261020-0` build (the directory
# containing `production/bin`); the script runs its `bmad_to_scibmad` program.
#
# The script exports a 1 m, 1 MV, 100 MHz rfcavity for a 1 GeV/c proton with Bmad `phi0` = 0, 1 and
# 0.25 using Bmad's `bmad_to_scibmad` program, tracks a reference particle through each emitted
# lattice and prints its relative momentum deviation after the cavity. `phi0` = 0 and 1 describe the
# same cavity, so their results must agree; Tao gives 0 for both and 1.371×10⁻³ for 0.25.
using SciBmad
haskey(ENV, "BMAD_SOURCE") || error("set BMAD_SOURCE to the root of the bmad-ecosystem 20261020-0 build (the directory that contains production/bin)")
const BIN = joinpath(ENV["BMAD_SOURCE"], "production", "bin", "bmad_to_scibmad")
isfile(BIN) || error("bmad_to_scibmad not found at $BIN")
dir = mktempdir()
lat(phi0) = """
parameter[particle] = proton
parameter[p0c] = 1e9
parameter[geometry] = open
beginning[beta_a] = 10
beginning[beta_b] = 10
rf: rfcavity, l = 1, voltage = 1e6, rf_frequency = 1e8, phi0 = $phi0
lat: line = (rf)
use, lat
"""
function export_and_track(tag, phi0)
    b = joinpath(dir, "$tag.bmad"); j = joinpath(dir, "$tag.jl")
    write(b, lat(phi0))
    run(pipeline(Cmd(`$BIN $b $j`; dir=dir, env=merge(ENV, Dict("OMP_NUM_THREADS" => "1"))), stdout=devnull, stderr=devnull))
    src = read(j, String)
    m = match(r"phi0 =\s*([-+0-9.Ee]+)", src)
    # The emitted file starts with `using Beamlines; using BeamTracking`; SciBmad re-exports both.
    mod = Module(); Core.eval(mod, :(using SciBmad))
    Base.include_string(mod, replace(src, r"^using .*$"m => ""))
    bl = Core.eval(mod, :lat).branches[1].beamlines[1]
    pz = track(bl; v0=zeros(1, 6)).v[1, 6, end]
    return (isnothing(m) ? "absent" : m[1]), pz
end
# Bmad reference values from Tao `show lat -orbit` on the same lattices (pz at the cavity exit):
# phi0 = 0 -> 0.0, phi0 = 1 -> 0.0, phi0 = 0.25 -> 1.370820E-03.
# Bmad documents phi0 in rad/2pi and tracks with 2pi*phi0; Beamlines documents phi0 in rad. The writer copies
# the value unscaled.
res = Dict(t => export_and_track(t, x) for (t, x) in (("p0", "0"), ("p1", "1"), ("pq", "0.25")))
for t in ("p0", "p1", "pq")
    println("Bmad phi0 = ", Dict("p0" => 0, "p1" => 1, "pq" => 0.25)[t], ": emitted phi0 = ", res[t][1], ", SciBmad pz after cavity = ", res[t][2])
end
println("expected: Bmad phi0 = 0 and 1 are the same cavity (period 1 in rad/2pi), so their SciBmad results must be equal;")
println("          Bmad phi0 = 0.25 gives pz = +1.370820E-03; the emitted 0.25 rad gives the value above")
println(res["p0"][2] != res["p1"][2] ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
