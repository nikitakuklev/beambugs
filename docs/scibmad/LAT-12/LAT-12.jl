# LAT-12: `writebl` fails for any electric multipole
#
# `writebl`, the Beamlines function that writes a lattice as Julia code and is documented as
# internal and under development, raises a `FieldError` for any element with an electric multipole,
# because it reads a field `tilt` where the electric multipole field is named `etilt`. The electric
# multipole's own `show` uses `etilt`, and the magnetic branch, which reads its own `tilt` field,
# writes `Kn1=0.2,tilt1=0.1` for a control element. An element with `En0 = 1` fails instead of
# producing text containing `En0=1.0`. Only lattices with electric multipoles are affected, and the
# failure is loud.
#
# The script calls `writebl` on a Beamline holding one element with `En0 = 1.0` and expects a
# `Beamline` expression containing `En0=1.0`. It also prints the field names of the electric
# multipole, and writes a magnetic element with `Kn1` and `tilt1` as a control.
using SciBmad
bl = Beamline([LineElement(L=1.0, En0=1.0)], species_ref=Species("proton"), pc_ref=1e9)
# writebl reads `em.tilt` from each electric multipole, whose field is named `etilt`.
r = try
    io = IOBuffer(); Beamlines.writebl(io, bl); String(take!(io))
catch e
    "ERROR: " * first(split(sprint(showerror, e), '\n'))
end
println("observed writebl(En0 element) -> ", r)
em = first(bl.line[1].EMultipoleParams)
println("EMultipole item properties: ", propertynames(em))
blm = Beamline([LineElement(L=1.0, Kn1=0.2, tilt1=0.1)], species_ref=Species("proton"), pc_ref=1e9)
io = IOBuffer(); Beamlines.writebl(io, blm); println("control magnetic writebl -> ", String(take!(io)))
println("expected: a Julia Beamline expression containing En0=1.0")
println(startswith(r, "ERROR") ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
