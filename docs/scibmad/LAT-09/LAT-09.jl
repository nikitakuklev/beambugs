# LAT-09: `track(...; use_KA=true)` on CPU errors; SIMD default ignores the override
#
# On the CPU, `track(bl; v0, use_KA=true)`, which selects KernelAbstractions (KA) kernels, raises
# “Cannot use both KernelAbstractions (KA) and explicit SIMD”, because `use_explicit_SIMD` keeps its
# CPU default `true` instead of following `use_KA`. The `track` docstring gives the default of
# `use_explicit_SIMD` as `!use_KA`, and both `TrackingConfig(use_KA=true)` and the lower-level
# `launch!` apply that rule. The single-keyword call fails, while adding `use_explicit_SIMD=false`
# or passing `config=TrackingConfig(use_KA=true)` returns identical coordinates. Only the keyword
# path of `track` is affected, and the failure is loud.
#
# The script tracks one electron through a 1.5 m drift with `use_KA=true` alone, with `use_KA=true`
# and `use_explicit_SIMD=false`, with `config=TrackingConfig(use_KA=true)` and with the defaults.
# The single-keyword call should behave as the explicit pair.
using SciBmad
bl = Beamline([Drift(L=1.5)], species_ref=Species("electron"), E_ref=1e9)
v0 = [1e-3 2e-3 -1e-3 1e-3 0 0]
function run(; kw...)
    try
        return track(bl; v0=v0, kw...).v[1, :, end]
    catch e
        return "ERROR: " * first(split(sprint(showerror, e), '\n'))
    end
end
# The CPU default config has use_KA=false and hence use_explicit_SIMD=true; each track keyword defaults
# independently from that config, so use_KA=true alone keeps explicit SIMD on and the kernel launch rejects both.
a = run(use_KA=true)
b = run(use_KA=true, use_explicit_SIMD=false)
c = run(config=TrackingConfig(use_KA=true))
d = run()
println("track(bl; v0, use_KA=true)                          -> ", a)
println("track(bl; v0, use_KA=true, use_explicit_SIMD=false) -> ", b)
println("track(bl; v0, config=TrackingConfig(use_KA=true))   -> ", c)
println("track(bl; v0) default                               -> ", d)
println("expected: use_KA=true alone behaves as use_KA=true, use_explicit_SIMD=!use_KA=false")
println(a isa String && !(b isa String) ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
