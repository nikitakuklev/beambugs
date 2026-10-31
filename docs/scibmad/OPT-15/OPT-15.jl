# OPT-15: Allocating `newton`/`brent` document caller mutation they do not do
#
# The docstrings of BatchSolve's allocating `newton` and `brent` say that the returned solution `u`
# is the caller's `x`, mutated in place, but both work on a copy and leave `x` unchanged. The
# in-place `brent!` does alias its argument, so the text describes the in-place functions. Calling
# `newton` from x = [2.0] returns u = [1.000] and leaves x = [2.0], with `u === x` false. Code that
# follows the docstring and reads `x` after the call gets the initial guess. The allocating behavior
# itself is sensible; only the documentation is wrong.
#
# The script calls `newton` on v² − 1 from x = [2.0] and `brent` on (v − 0.25)² from x = [0.0], and
# checks whether the returned `u` is the caller's `x` and whether `x` changed. As a control, the
# in-place `brent!` on the same minimization returns its argument, which then holds the solution.
using SciBmad
const BS = SciBmad.BatchSolve
x = [2.0]
# Both allocating wrappers pass copy(x) to the in-place solver.
s = BS.newton(v -> v .^ 2 .- 1, x)
println("newton: u=", s.u, "  x after call=", x, "  u === x: ", s.u === x)
xb = [0.0]
t = BS.brent(v -> (v .- 0.25) .^ 2, xb; xa = [-1.0], xb = [1.0])
println("brent : u=", t.u, "  x after call=", xb, "  u === x: ", t.u === xb)
xi = [0.0]; yi = similar(xi)
ti = BS.brent!((y, v) -> (y .= (v .- 0.25) .^ 2; y), yi, xi; xa = [-1.0], xb = [1.0])
println("independent: brent! u === x: ", ti.u === xi, "  x after call=", xi)
println(!(s.u === x) && !(t.u === xb) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
