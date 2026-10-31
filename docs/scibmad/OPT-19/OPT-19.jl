# OPT-19: Reading a monomial the descriptor does not hold terminates the process
#
# In GTPSA, reading the coefficient of a monomial that the descriptor does not hold ends the Julia
# process from the C library with the message `invalid monomial`, with no Julia exception or stack
# trace; code in `catch`, `finally` and `atexit` never runs. An out-of-range indexing call should
# throw a catchable exception, and GTPSA's own validity test `mad_desc_isvalidm` already identifies
# such monomials. For t = (x + y)² under `Descriptor([1,1,1], 2)`, whose per-variable caps of 1
# exclude every square, `t[[1,1,0]]` correctly returns 2.000, but `t[[2,0,0]]` ends the session with
# exit status 1. Per-variable caps, such as those of the default `twiss(chrom=2)` descriptor, make
# such reads easy to issue; by source reading, the monomial setter lacks the same check.
#
# The script builds t = (x + y)² under `Descriptor([1,1,1], 2)`, asks `mad_desc_isvalidm` about
# [1,1,0] and [2,0,0], reads the valid coefficient as a control, then reads `t[[2,0,0]]` inside
# `try`/`catch`. Because the library ends the process with C `exit`, a C-level exit handler
# registered with `__cxa_atexit` prints the verdict while the read is pending, and a Julia `atexit`
# hook that stays silent shows that Julia never unwinds. Run it alone, as its own process: nothing
# after the invalid read survives, and it exits with status 1.
using SciBmad
# The library ends the process with C exit(1), which skips Julia's own atexit hooks and any
# try/catch, so this C-level handler prints the verdict, but only while the invalid read is pending.
const PENDING = Ref(false)
onexit(_) = (PENDING[] && ccall(:puts, Cint, (Cstring,), "DEFECT REPRODUCED (process terminated inside t[[2,0,0]])"); nothing)
ccall((:__cxa_atexit, "libc.so.6"), Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), @cfunction(onexit, Cvoid, (Ptr{Cvoid},)), C_NULL, C_NULL)
try_catch_saw = Ref("nothing")
atexit(() -> PENDING[] && println("julia atexit hook ran"))   # independent: does Julia unwind?
d = Descriptor([1, 1, 1], 2); x = vars(d)
t = (x[1] + x[2])^2
okm = Cuchar[1, 1, 0]; badm = Cuchar[2, 0, 0]
println("independent: mad_desc_isvalidm([1,1,0]) = ", GTPSA.mad_desc_isvalidm(d.desc, Cint(3), okm),
        "  mad_desc_isvalidm([2,0,0]) = ", GTPSA.mad_desc_isvalidm(d.desc, Cint(3), badm))
println("control: t[[1,1,0]] = ", t[[1, 1, 0]], " (expected 2.0, the x*y coefficient of (x+y)^2)")
println("expected: t[[2,0,0]] throws a Julia exception (or returns 0.0)")
flush(stdout)
PENDING[] = true
v = try
    t[[2, 0, 0]]
catch e
    try_catch_saw[] = string(nameof(typeof(e))); nothing
end
PENDING[] = false
println("t[[2,0,0]] returned ", v, "; exception caught: ", try_catch_saw[])
println("NOT REPRODUCED")
