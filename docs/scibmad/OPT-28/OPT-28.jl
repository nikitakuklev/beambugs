# OPT-28: `I - map` retains off-diagonal, nonlinear and spin coefficient signs
#
# For a NonlinearNormalForm `DAMap`, `TPSAMap` or `VectorField`, `I - M`, with `I` the identity,
# replaces only the diagonal linear coefficients and the scalar quaternion coefficient by one minus
# their values, and every other coefficient keeps its original sign. Coefficient subtraction must
# satisfy (I − M) + M = I, and ordinary subtraction `one(M) - M` gives the analytic result. For the
# shear M = (x, p − 2.5x + 0.75x²), the second component of `I - M` has x and x² coefficients −2.5
# and 0.75 instead of 2.5 and −0.75. A spin coefficient q1[x] = 0.2 stays 0.2 instead of becoming
# −0.2. `M - I` is correct, and no effect on optics outputs has been established.
#
# The script builds the shear as a `DAMap` and as a `TPSAMap` on `Descriptor(2, 3)`, compares the x
# and x² coefficients of the second component of `I - M` with those of `one(M) - M`, and prints the
# coefficient norm of (I − M) + M − `one(M)`. It also subtracts from `I` a `VectorField` with spin
# coefficient q1[x] = 0.2; `M - I` against `M - one(M)` is a passing control.
using SciBmad, LinearAlgebra
const NNF = NonlinearNormalForm
BLAS.set_num_threads(1)
x,p = vars(Descriptor(2,3))
checks = Bool[]
for kind in (DAMap, NNF.TPSAMap)
    M = kind(nv=2,v=[x,p-2.5x+0.75x^2])
    bad = I-M
    correct = one(M)-M
    println(kind," (I-M).p coefficients x,x^2 = ",(bad.v[2][[1,0]], bad.v[2][[2,0]])," expected = ",(correct.v[2][[1,0]], correct.v[2][[2,0]]))
    r = (I-M)+M-one(M)
    println("algebraic identity residual = ",maximum(GTPSA.normTPS.(r.v)))
    @assert correct.v[2][[1,0]] == 2.5
    push!(checks, bad.v[2][[1,0]] == -2.5)
    @assert correct.v[2][[2,0]] == -0.75
    push!(checks, bad.v[2][[2,0]] == 0.75)
    # M-I is a correct independent control.
    @assert maximum(GTPSA.normTPS.((M-I).v .- (M-one(M)).v)) == 0
end
F = VectorField(v=[zero(x),zero(p)],q=[zero(x),0.2x,zero(x),zero(x)])
println("(I-F).q1[x] = ",(I-F).q.q1[[1,0]]," expected = -0.2")
push!(checks, (I-F).q.q1[[1,0]] == 0.2)
println("independent: ordinary map subtraction and M-I controls pass exactly")

println(all(checks) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
