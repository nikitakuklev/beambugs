# LAT-10: Exported `ParticleView` cannot be constructed
#
# Constructing the exported `ParticleView(bunch, i)`, meant as a view of particle i of a `Bunch`,
# fails with “too many parameters for type ParticleView”: the struct declares six type parameters,
# but its inner constructor supplies eight, one per field, as reflection confirms. For a bunch whose
# particle 2 has coordinates [2, 5, 8, 11, 14, 17], the call raises instead of returning that row.
# Source reading shows a second fault behind the first: the outer constructor takes coordinate
# columns `view(v, :, i)` and `view(q, :, i)`, whereas particles are rows of the N×6 and N×4 arrays,
# so the view would hold wrong data once construction works. No code in the SciBmad packages uses
# `ParticleView`.
#
# The script builds a three-particle `Bunch` whose coordinates are 1 to 18 and calls
# `ParticleView(bunch, 2)`, expecting row 2 of the coordinate matrix. It also prints the numbers of
# type parameters and fields of `ParticleView`.
using SciBmad
const BT = SciBmad.BeamTracking
b = Bunch(v=reshape(collect(1.0:18.0), 3, 6))
# ParticleView declares six type parameters {B,T,S,V,Q,W}, but its inner constructor passes the types of all
# eight fields (index::Int and species::Species included).
r = try
    pv = ParticleView(b, 2)
    pv.v
catch e
    "ERROR: " * first(split(sprint(showerror, e), '\n'))
end
println("observed ParticleView(bunch, 2).v = ", r)
println("expected particle row 2 of the N x 6 matrix = ", b.v[2, :])
T = Base.unwrap_unionall(ParticleView)
println("type parameters = ", length(T.parameters), ", fields = ", fieldcount(T))
println(r isa String ? "DEFECT REPRODUCED" : (r == b.v[2, :] ? "NOT REPRODUCED" : "DEFECT REPRODUCED"))
