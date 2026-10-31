# LAT-23: `BatchParam` scalar shortcuts test `≈ 0` and `≈ 1`, dropping factors, derivatives and
# knobs
#
# Arithmetic between a scalar and an array-backed `BatchParam` (addition, subtraction,
# multiplication and batch/scalar division) takes shortcuts when the scalar is approximately 0 or 1,
# discarding derivatives and power-series terms. ForwardDiff multipliers at 1 lose their
# derivatives, zero cases depend on the signs of the zero and the derivative, and GTPSA series with
# scalar part 0 or 1 lose their nonconstant coefficients. For a 0.5 m quadrupole with `Kn1 = s *
# BatchParam([0.3, 0.6])` and initial x = 1 mm, dx/ds at s = 1 is [0, 0] instead of [−1.833×10⁻⁴,
# −3.583×10⁻⁴] m, which scalar tracking and finite differences recover. Float64 factors within a
# relative 1.490×10⁻⁸ of 1 and Float32 factors within 3.453×10⁻⁴ are ignored, and division by zero
# gives +∞ in every lane.
#
# With the batch b = [1, −2, 0.5], the script checks (a) a Float64 factor 1 + 10⁻⁹ and division by
# zero, (b) ForwardDiff derivatives of the lanes at scalar values 0 and 1 against central
# differences, and (c) a GTPSA knob added to or multiplying the batch against the same operation
# applied lane by lane. It then (d) differentiates the final x of a tracked quadrupole with `Kn1 = s
# * BatchParam([0.3, 0.6])` at s = 1 and compares with central differences and with per-lane scalar
# tracking.
using SciBmad
using SciBmad: ForwardDiff
using SciBmad.GTPSA
const D = ForwardDiff.Dual
b = [1.0, -2.0, 0.5]; bp = BatchParam(b)
lanes(r) = r.batch isa Number ? fill(r.batch, length(b)) : collect(r.batch)
bad = false

# A scalar operand ≈ 0 or ≈ 1 returns the batch unchanged (or BatchParam(0f0) / BatchParam(Inf32)). isapprox
# admits a relative difference up to sqrt(eps) = 1.49e-8; a Dual with value 1 is ≈ 1 whatever its partials, and
# a Dual with value ±0 is ≈ 0 for one sign of its partial (|x - y| <= rtol*max(|x|, |y|), with ForwardDiff
# ordering equal values by their partials); GTPSA's isapprox compares only the scalar part of a TPS.
println("(a) Float factor 1 + 1e-9: lanes unchanged?  expected lane-wise b .* f (every lane scaled)")
f = 1 + 1e-9
for (name, r, e) in (("bp * f", bp * f, b .* f), ("f * bp", f * bp, f .* b), ("bp / f", bp / f, b ./ f))
    global bad |= lanes(r) != e
    println("  ", rpad(name, 8), "observed ", lanes(r), " == b: ", lanes(r) == b, "; lane-wise ", e, " == b: ", e == b)
end
println("  bp / 0 = ", lanes(bp / 0), "; lane-wise b ./ 0 = ", b ./ 0)

println("(b) ForwardDiff derivative of the lanes; recheck: central difference, h = 1e-6 (above the 1.5e-8 window)")
h = 1e-6
for (name, g, x0) in (("d * bp", d -> d * bp, 1.0), ("bp / d", d -> bp / d, 1.0),
                      ("(1 - d) * bp", d -> (1 - d) * bp, 1.0), ("(0 - d) + bp", d -> (0 - d) + bp, 0.0),
                      ("(-d) + bp (control)", d -> (-d) + bp, 0.0), ("d * bp (control)", d -> d * bp, 0.0))
    ad = ForwardDiff.derivative(d -> lanes(g(d)), x0)
    fd = (lanes(g(x0 + h)) - lanes(g(x0 - h))) / 2h
    err = maximum(abs, ad - fd)
    occursin("control", name) || (global bad |= err > 1e-6)
    println("  ", rpad(name * " at d = $x0", 32), "ForwardDiff ", ad, "  central difference ", round.(fd; digits=9), "  max|diff| ", err)
end
println("  ≈ itself: Dual(1,±1) ≈ 1 ", D(1.0, 1.0) ≈ 1, " ", D(1.0, -1.0) ≈ 1, "; Dual(+0,-1) ≈ 0 ", D(0.0, -1.0) ≈ 0,
        ", Dual(-0,+1) ≈ 0 ", D(-0.0, 1.0) ≈ 0, ", Dual(+0,+1) ≈ 0 ", D(0.0, 1.0) ≈ 0)

println("(c) GTPSA knob x = 0 + 1·dx: first-order coefficient of each lane; recheck: the same operation lane by lane")
x = vars(Descriptor(1, 1))[1]
coef(t) = t isa TPS ? t[1] : 0.0
for (name, g) in (("x + bp", (x, s) -> x + s), ("x * bp", (x, s) -> x * s), ("(1 + x) * bp", (x, s) -> (1 + x) * s),
                  ("bp / (1 + x)", (x, s) -> s / (1 + x)))
    r = lanes(g(x, bp))
    e = [coef(g(x, bi)) for bi in b]
    global bad |= map(coef, r) != e
    println("  ", rpad(name, 14), "observed eltype ", eltype(r), ", d/dx ", map(coef, r), "  lane by lane ", e)
end
println("  x ≈ 0: ", x ≈ 0, ", isone(1 + x): ", isone(1 + x), ", isequal(x, 0): ", isequal(x, 0))

println("(d) tracking: Kn1 = s * BatchParam([0.3, 0.6]); d x_final / ds at s = 1")
function xfinal(s, K)
    ele = Quadrupole(L=0.5, Kn1=s * K)
    bl = Beamline([ele, Drift(L=1.0)], species_ref=Species("electron"), E_ref=1e9)
    n = K isa BatchParam ? 2 : 1
    bunch = Bunch(repeat([1e-3 0 0 0 0 0], n, 1) .+ zero(s); p_over_q_ref=bl.p_over_q_ref, species=bl.species_ref)
    track!(bunch, bl)
    return bunch.coords.v[:, 1]
end
K = BatchParam([0.3, 0.6])
adb = ForwardDiff.derivative(s -> xfinal(s, K), 1.0)
fdb = (xfinal(1 + h, K) - xfinal(1 - h, K)) / 2h
ads = [ForwardDiff.derivative(s -> xfinal(s, k)[1], 1.0) for k in (0.3, 0.6)]
global bad |= maximum(abs, adb - ads) > 1e-12
println("  batched ForwardDiff ", adb, "\n  batched central difference ", fdb,
        "\n  recheck, scalar Kn1 = s * K per lane, ForwardDiff ", ads)
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
