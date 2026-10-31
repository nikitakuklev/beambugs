# TRK-05: `Patch(dt)` shifts z but not `Bunch.t_ref`
#
# `Patch(dt)` shifts each particle's longitudinal coordinate z but leaves the bunch reference time
# `t_ref` unchanged, so the absolute particle time reconstructed from them jumps. A reference-time
# shift must advance `t_ref` by dt with z and preserve the particle's arrival time t, as the
# definition z = βc(`t_ref` − t) (β is speed over light speed c) and Bmad's patch implementation
# require. With dt = 2.000 ns and an initial `t_ref` of 30.00 ns, `t_ref` advances by 0 instead of
# 2.000 ns and the particle time changes by −2.000 ns instead of 0. Every later time-dependent
# element samples the wrong time: a dipole kick ramped at 3.000 × 10⁵ s⁻¹ changes by 6.000 × 10⁻⁴
# when the patch is inserted.
#
# The script tracks a proton with p₀c = 2mc², z = 0.025 m and relative momentum offset δ = 0.08
# through `Patch(dt = 2 ns)` from `t_ref` = 30 ns, and prints the changes of `t_ref`, z and the
# particle time t = `t_ref` − z/(βc), against the expected dt, βc·dt and 0. A recheck tracks the
# particle through a `Multipole` whose dipole kick ramps as 3 × 10⁵ s⁻¹ × `Time()`, with and without
# the patch in front; the kick should not change.
using SciBmad

const sp = Species("proton"); const pc = 2 * massof(sp)
const R = pc / (C_LIGHT * chargeof(sp))
v = [0.0 0.0 0.0 0.0 0.025 0.08]
beta = (1 + v[6]) / hypot(1 + v[6], massof(sp) / pc)
t0 = 3e-8; dt = 2e-9

b = Bunch(copy(v), species = sp, p_over_q_ref = R, t_ref = t0)
track!(b, Beamline([Patch(dt = dt)], species_ref = sp, pc_ref = pc); use_explicit_SIMD = false)
# A reference-clock shift advances t_ref by dt together with z += beta c dt, so the particle's
# absolute time t = t_ref - z/(beta c) must not change (Fortran Bmad does both).
t_before = t0 - v[5] / (beta * C_LIGHT)
t_after = b.t_ref - b.v[1, 5] / (beta * C_LIGHT)
println("t_ref change           = ", b.t_ref - t0, " s   (expected ", dt, ")")
println("z change               = ", b.v[1, 5] - v[5], " m   (expected beta*c*dt = ", beta * C_LIGHT * dt, ")")
println("particle time change   = ", t_after - t_before, " s   (expected 0)")
# --- Independent recheck: a Time()-dependent kicker after the patch sees the shifted clock ---
slope = 3e5
kick(line) = (bb = Bunch(copy(v), species = sp, p_over_q_ref = R, t_ref = t0);
              track!(bb, Beamline(line, species_ref = sp, pc_ref = pc); use_explicit_SIMD = false); bb.v[1, 2])
k0 = kick([Multipole(Kn0L = slope * Time())]); k1 = kick([Patch(dt = dt), Multipole(Kn0L = slope * Time())])
println("recheck: Kn0L = slope*Time() kick without patch = ", k0, ", after Patch(dt) = ", k1,
        ", difference = ", k1 - k0, " (slope*dt = ", slope * dt, "; expected 0)")
println(abs(t_after - t_before + dt) < 1e-15 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
