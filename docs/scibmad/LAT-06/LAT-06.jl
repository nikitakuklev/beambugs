# LAT-06: Multipole tilt assignment needing type promotion errors
#
# Assigning a multipole tilt that needs a wider number type than the element's stored strengths
# raises `Incorrect value for BMultipoleParams` when that multipole order already exists. `tilt1 =
# 0.1` on the Float32 `Quadrupole(Kn1=1f0)` fails, as does a ForwardDiff `Dual` tilt on an ordinary
# Float64 quadrupole, and the electric `etilt1` fails the same way. The assignment should store the
# promoted value, as it does for an order the element does not yet have (`tilt2 = 0.1` returns 0.1)
# and for an assignment that needs no promotion. Derivatives with respect to multipole tilts
# therefore cannot be taken by assigning a `Dual` tilt; the failure is loud and produces no wrong
# values.
#
# The script attempts three promoting tilt assignments to an existing order: `tilt1 = 0.1` on a
# Float32 quadrupole, a `Dual` `tilt1` on a Float64 quadrupole and `etilt1 = 0.1` on a Float32
# electric multipole. Two controls must succeed: a tilt for a new order on the Float32 quadrupole
# and a tilt that needs no promotion.
using SciBmad
const FD = SciBmad.ForwardDiff
function attempt(label, f, expected)
    try
        val = f()
        println(rpad(label, 44), " observed ", val, "  expected ", expected)
        return val == expected
    catch e
        println(rpad(label, 44), " observed ERROR: ", first(split(sprint(showerror, e), '\n')), "  expected ", expected)
        return false
    end
end
# For an existing order, param_replace returns the scalar from setproperty! instead of the promoted
# parameter group, and the element then stores that scalar as the group, which fails validation.
ok = Bool[]
push!(ok, attempt("Quadrupole(Kn1=1f0); tilt1 = 0.1", () -> (e = Quadrupole(Kn1=1f0); e.tilt1 = 0.1; e.tilt1), 0.1))
push!(ok, attempt("Quadrupole(Kn1=0.36); tilt1 = Dual(0.1,1)", () -> (e = Quadrupole(Kn1=0.36); d = FD.Dual(0.1, 1.0); e.tilt1 = d; e.tilt1 == d), true))
push!(ok, attempt("LineElement(En1=1f0); etilt1 = 0.1", () -> (e = LineElement(En1=1f0); e.etilt1 = 0.1; e.etilt1), 0.1))
println("controls (promotion through the add-new-order branch, bmultipole.jl:159-165):")
push!(ok, attempt("Quadrupole(Kn1=1f0); tilt2 = 0.1 (new order)", () -> (e = Quadrupole(Kn1=1f0); e.tilt2 = 0.1; e.tilt2), 0.1))
push!(ok, attempt("Quadrupole(Kn1=1.0); tilt1 = 0.1 (no promotion)", () -> (e = Quadrupole(Kn1=1.0); e.tilt1 = 0.1; e.tilt1), 0.1))
println(all(ok[4:5]) && !all(ok[1:3]) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
