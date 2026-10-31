# LAT-24: A ForwardDiff or GTPSA scalar combined with a Float32 `BatchParam` throws
#
# Combining a ForwardDiff `Dual` or a GTPSA power-series scalar with a Float32 array-backed
# `BatchParam` fails when the operation reaches scalar conversion: `Dual(2.0, 1.0) *
# BatchParam(Float32[1, -2])` raises a `Float32(::Dual)` method error, and a power series with
# constant part 2 fails conversion to Float32 the same way. The conversion, meant to preserve lane
# precision, is wrongly applied to these richer scalar types. The same `Dual` with a Float64 batch
# gives values [2, −4] and derivatives [1, −2], the lane-wise result. Single-precision batch
# differentiation through this path fails before tracking on the CPU; the approximate-zero/one
# shortcuts instead bypass the conversion and silently lose derivative terms.
#
# The script combines the batch `Float32[1, -2]` with `Dual(2.0, 1.0)` by multiplication and
# addition, with `Dual(2f0, 1f0)` and with the GTPSA series 2 + x, and builds a quadrupole from such
# an expression; each should give the lane-wise `Dual` or series values. The controls are a Float64
# scalar times the Float32 batch and the same `Dual` with a Float64 batch.
using SciBmad
using SciBmad: ForwardDiff
using SciBmad.GTPSA
const D = ForwardDiff.Dual
x = vars(Descriptor(1, 1))[1]
b32 = Float32[1, -2]; bp32 = BatchParam(b32)
attempt(f) = try
    f()
catch e
    "ERROR " * first(split(sprint(showerror, e), '\n'))
end
# For a Float32 or Float16 batch, every scalar-batch operation converts the scalar with eltype(batch).(scalar),
# which has no method for a Dual or a TPS.
bad = false
for (name, s, op) in (("Dual(2.0, 1.0) * bp32", D(2.0, 1.0), *), ("Dual(2.0, 1.0) + bp32", D(2.0, 1.0), +),
                      ("Dual(2f0, 1f0) * bp32", D(2f0, 1f0), *), ("(2 + x) * bp32", 2 + x, *))
    r = attempt(() -> op(s, bp32).batch)
    e = map(bi -> op(s, bi), b32)                     # lane-wise scalar arithmetic
    global bad |= r isa String
    println(rpad(name, 24), "observed ", r, "\n", " "^24, "expected lane-wise ", e isa Vector{<:TPS} ? map(t -> (t[0], t[1]), e) : e)
end
println("controls: 2.0 * bp32 = ", (2.0 * bp32).batch, " (", eltype((2.0 * bp32).batch), ", the intended lowering)",
        "; Dual(2.0, 1.0) * BatchParam(Float64[1, -2]) = ", (D(2.0, 1.0) * BatchParam([1.0, -2.0])).batch)
println("element: Quadrupole(L=0.5, Kn1=Dual(2.0, 1.0) * bp32): ", attempt(() -> (Quadrupole(L=0.5, Kn1=D(2.0, 1.0) * bp32); "built")))
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
