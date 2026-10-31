# OPT-14: Batched Newton returns the previous iterate's `f` and `jac`
#
# BatchSolve's batched `newton` returns stale `f` and `jac` for lanes (problems in the batch) that
# the relative-step test accepts in the final iteration: it updates the iterate, then exits before
# recomputing the promised final residual and Jacobian. For x² − 2 from x = 1 with `reltol = 0.4`,
# the accepted iterate is 1.5, but the returned residual and Jacobian are −1.000 and 2.000 instead
# of 0.2500 and 3.000; the non-batched solver returns the latter values. The accepted iterate is
# unchanged by this diagnostic defect, and residual-test exits are unaffected. By source inspection,
# batched closed-orbit finding uses this path and experimental aperture code reads the Jacobian as I
# − M, with M the one-turn matrix; aperture effects were not measured.
#
# The script solves f(x) = x² − 2 from x = 1 with `reltol = 0.4`, once with the batched solver (two
# lanes) and once with the non-batched solver as the control. One step gives x = 1.5, and |Δx| = 0.5
# < 0.4·1.5 ends the iteration, so the returned `f` and `jac` must equal f(1.5) = 0.25 and f′(1.5) =
# 3.
using SciBmad
const BS = SciBmad.BatchSolve
f(x) = x .^ 2 .- 2
# In the batched branch x is updated, the relative-step test marks the lane converged, and the loop
# breaks before the residual and Jacobian are evaluated again; the non-batched branch re-evaluates
# before its step test.
b = BS.newton(f, fill(1.0, 2, 1); batchdim = 1, reltol = 0.4, abstol = 1e-14)
n = BS.newton(f, [1.0]; reltol = 0.4, abstol = 1e-14)
println("batched    : u=", vec(b.u), " f=", vec(b.f), " jac=", Matrix(b.jac), " retcode=", vec(b.retcode), " iters=", vec(b.iters))
println("expected   : f(u)=", vec(f(b.u)), " jac diag=", 2 .* vec(b.u))
println("non-batched: u=", n.u, " f=", n.f, " jac=", Matrix(n.jac), " retcode=", n.retcode, " iters=", n.iters)
println(maximum(abs, b.f .- f(b.u)) > 1e-12 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
