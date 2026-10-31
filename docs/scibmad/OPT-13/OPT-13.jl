# OPT-13: Out-of-place `F * m`, `a * m`, `lb`, mixed-type `exp(F, m)` and `VectorField(F; init)` are
# unusable
#
# Several out-of-place (allocating) operations on NonlinearNormalForm maps and vector fields throw
# instead of computing. `F * m` (a vector field acting on a `DAMap`) and `2.0 * m` are ambiguous
# method calls, the Lie bracket `lb(F, H)` references an undefined `G`, `exp(F, m)` with a real
# field and a complex map reads an undefined `m1prom`, and the copy constructor `VectorField(F)`
# reads an undefined `V0`. On the same inputs, the in-place kernels `mul!` and `lb!` give the
# analytic results, `exp` with both arguments complex runs, and `Test.detect_ambiguities` lists
# exactly the two `*` ambiguities. Every failure is a loud error; no wrong value is returned.
#
# The script builds the identity `DAMap` m and the vector fields F = (p, −x) and H = (x², 0) on a
# `Descriptor(2, 3)`, calls each out-of-place form and prints the first line of its error. Beside
# each it runs the working counterpart: `mul!` against F·∇id = (p, −x), `lb!` against [F,H] = (2xp,
# x²), `exp` with both arguments complex, and `Test.detect_ambiguities` on the package for the two
# `*` ambiguities.
using SciBmad
const NNF = NonlinearNormalForm
d = Descriptor(2, 3); x, p = vars(d); z = zero(x)
F = VectorField(v = [p, -x]); H = VectorField(v = [x^2, z])
m0 = DAMap(v = [x, p], nv = 2)
res(f) = try f(); "ok" catch err; first(split(sprint(showerror, err), '\n')) end
# (a) F * m is ambiguous with *(m2, m1::DAMap) = m2 ∘ m1; (c) that method also makes Number * DAMap
# ambiguous with *(::Number, ::Union{TaylorMap,VectorField}).
println("(a) F * m0       -> ", res(() -> F * m0))
r = zero(m0); NNF.mul!(r, F, m0)
println("    in-place mul!: |(F*m0).v - (p, -x)| = ", GTPSA.normTPS(r.v[1] - p) + GTPSA.normTPS(r.v[2] + x))
using Test
amb = filter(t -> t[1].name == :* && t[2].name == :* && occursin(r"VectorField|Number", string(t[1].sig) * string(t[2].sig)) &&
                  occursin("DAMap", string(t[1].sig) * string(t[2].sig)), Test.detect_ambiguities(NNF))
println("    Test.detect_ambiguities(NNF), pairs of `*`: ", [(string(a.sig), string(b.sig)) for (a, b) in amb])
println("(c) 2.0 * m0     -> ", res(() -> 2.0 * m0))
# (b) lb(F, H) calls promote(F, G) with no G in scope, then overwrites its argument H.
println("(b) NNF.lb(F, H) -> ", res(() -> NNF.lb(F, H)))
G = zero(F); NNF.lb!(G, F, H)
# F·∇H - H·∇F with F=(p,-x), H=(x²,0): (2xp - 0, 0 + x²)
println("    in-place lb!:  |[F,H].v - (2xp, x²)| = ", GTPSA.normTPS(G.v[1] - 2x*p) + GTPSA.normTPS(G.v[2] - x^2))
# (d) With a real F and a complex m, exp(F, m) promotes only F and then reads m1prom, which only the
# other branches set; (e) VectorField(F; init) reads an undefined V0.
println("(d) exp(F, complex(m0)) -> ", res(() -> exp(F, complex(m0))))
println("    both complex:  exp(complex F, complex(m0)) -> ", res(() -> exp(complex(typeof(F))(F), complex(m0))))
println("(e) VectorField(F) -> ", res(() -> VectorField(F)))
println(occursin("ambiguous", res(() -> F * m0)) && occursin("G", res(() -> NNF.lb(F, H))) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
