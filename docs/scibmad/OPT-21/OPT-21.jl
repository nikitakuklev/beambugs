# OPT-21: `getterm(…, as_taylor_series=true)` drops the constant coefficient
#
# `getterm(q; as_taylor_series=true)` with all exponents zero, which returns a tune, `etac` or
# `alphac` as a Taylor series in the knobs (parameters such as quadrupole strengths), keeps the
# dependence on the knobs but drops the constant term. That constant must equal the scalar `getterm`
# result, but the code starts GTPSA's coefficient iterator at 0, which skips the constant; starting
# at −1 includes it. For the quadrupole-knob cell of SciBmad's parametric example, the constant of
# `q1` is 0.0 instead of 0.1436, while the knob gradients 0.2738 and 0.1217 are correct. Tune
# matching through this call sees the tune shifts but not the tune; requests with any nonzero
# exponent, such as the documented `delta=1`, are unaffected.
#
# The script builds the two-thin-quadrupole cell of SciBmad's parametric normal-form example, with
# knobs on both integrated strengths (`Descriptor([1,1,1,1,1,1], 2, [1,1], 1)`), and compares the
# Taylor constant and knob gradients of `q1` with the tune and its gradient from thin-lens matrices.
# The scalar `getterm`, the printed value and the `delta=1` request are controls, and the iterator
# itself is run on the constant TPS 2 from start indices 0 and −1.
using SciBmad, LinearAlgebra
const FD = SciBmad.ForwardDiff
d = Drift(L = 1.0)
kq = Beamline([Multipole(Kn1L = DefExpr(c -> c.kf)), d, Multipole(Kn1L = DefExpr(c -> c.kd)), d],
              species_ref = Species("electron"), pc_ref = 1e9)
desc = Descriptor([1, 1, 1, 1, 1, 1], 2, [1, 1], 1); dk = params(desc)
kq.context.kf = 0.8 + dk[1]; kq.context.kd = -0.7 + dk[2]
tk = twiss(kq, GTPSA_descriptor = desc, at = [], rf_on = false)
qa(k) = acos(tr([1.0 1; 0 1] * [1.0 0; -k[2] 1] * [1.0 1; 0 1] * [1.0 0; -k[1] 1]) / 2) / 2pi
k0 = [0.8, -0.7]
println("analytic: q1 = ", qa(k0), "  dq1/dk = ", FD.gradient(qa, k0))
ts = getterm(tk.q1, as_taylor_series = true)
println("getterm(q1, as_taylor_series=true): constant = ", scalar(ts), "  dq1/dk = ",
        [ts[[0, 0, 0, 0, 0, 0, 1, 0]], ts[[0, 0, 0, 0, 0, 0, 0, 1]]])
println("independent: getterm(q1) = ", getterm(tk.q1), "  printed: ", strip(SciBmad._show_adv(tk.q1)))
println("control: getterm(q1, delta=1) = ", getterm(tk.q1, delta = 1), "  Taylor constant = ",
        scalar(getterm(tk.q1, delta = 1, as_taylor_series = true)))
# GTPSA's iterator returns the next nonzero index after the one given; −1 is the documented start
# that includes index 0, the constant.
t = 2 + TPS64(use = desc); v = Ref(0.0)
println("independent: constant TPS 2: cycle!(t, 0) = ", SciBmad.TI.cycle!(t, 0, val = v),
        "  cycle!(t, -1) = ", SciBmad.TI.cycle!(t, -1, val = v), " with value ", v[])
bad = scalar(ts) == 0.0 && abs(getterm(tk.q1) - qa(k0)) < 1e-14
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
