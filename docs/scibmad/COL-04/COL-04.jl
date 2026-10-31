# COL-04: `twiss(spin=true)` fails when the spin axis is exactly vertical
#
# `twiss(spin=true)` fails for a ring whose closed-orbit spin axis is exactly vertical, because the
# normal-form step that aligns this axis divides by zero. In a flat ring the one-turn spin motion is
# a regular rotation about the vertical by 2π·aγ, where a is the magnetic anomaly and γ the Lorentz
# factor, and directly tracked spin quaternions agree with that analytic rotation. For electrons
# with γ = 100.0 in a ring of 40 identical combined-function bends, spin Twiss throws `InexactError`
# instead of returning a spin tune of magnitude 0.1160. Orbital Twiss succeeds, and tilting one bend
# by 1.000 × 10⁻⁹ rad makes spin Twiss succeed; roundoff-sized departures from vertical can also
# hide the failure.
#
# The script builds a flat ring of 40 identical combined-function `SBend`s for electrons at γ = 100,
# tracks one turn with spin and compares the one-turn quaternion with the analytic rotation by 2π·aγ
# about the vertical. It then runs orbital Twiss, which succeeds, and spin Twiss, which throws. As a
# control, it tilts one bend by 10⁻⁹ rad and checks that spin Twiss then returns the spin tune −aγ
# modulo 1.
using SciBmad, LinearAlgebra

sp = Species("electron"); gam = 100.0; m = massof(sp); a = gyromagnetic_anomaly(sp)
b = SBend(L = 2pi * 10 / 40, angle = 2pi / 40, Kn1 = -0.003,
          tracking_method = Symplectic(order = 4, n_steps = 4, fringe_at = Fringe.NoEnd))
ring = Beamline(fill(b, 40), species_ref = sp, E_ref = gam * m)

r = track(ring, spin = true, use_KA = false, use_explicit_SIMD = false)
q = vec(r.q[1, :, end])
println("tracked one-turn quaternion = ", q)
println("analytic (rotation 2*pi*a*gamma about y) = ", [cos(pi * a * gam), 0.0, -sin(pi * a * gam), 0.0])
println("exact zeros in q_x, q_z: ", q[2] == 0.0 && q[4] == 0.0)
println("expected spin tune magnitude frac(a*gamma) = ", mod(a * gam, 1))
tw0 = twiss(ring, rf_on = false, spin = false, at = [0])
println("orbital twiss succeeds: ", join(["$k=$(tw0.summ.sdict[k])" for k in keys(tw0.summ.sdict)], ", "))
# NonlinearNormalForm.normal aligns n0 with y using nrm = sqrt(n0x^2 + n0z^2) as a divisor, which
# is 0 for n0 = (0, +-1, 0). twiss extracts the spin tune as -atan(q2, q0)/pi, so the expected
# result is +-frac(a*gamma).
failed = false
try
    tw = twiss(ring, rf_on = false, spin = true, at = [0])
    println("spin twiss: ", join(["$k=$(tw.summ.sdict[k])" for k in keys(tw.summ.sdict)], ", "))
catch err
    global failed = true
    st = stacktrace(catch_backtrace())
    k = findfirst(f -> occursin("normal.jl", string(f.file)), st)
    println("spin twiss: ", typeof(err), ": ", sprint(showerror, err)[1:min(end, 80)])
    isnothing(k) || println("  thrown from ", st[k].func, " at ", basename(string(st[k].file)), ":", st[k].line)
end
# Independent recheck: tilt one of the 40 bends by 1e-9 rad. n0 moves off exact vertical by
# O(1e-9), the physics is unchanged to that order, and normal() no longer divides by zero.
mk(tl) = SBend(L = 2pi * 10 / 40, angle = 2pi / 40, Kn1 = -0.003, tilt_ref = tl,
               tracking_method = Symplectic(order = 4, n_steps = 4, fringe_at = Fringe.NoEnd))
ring2 = Beamline([mk(1e-9); fill(mk(0.0), 39)], species_ref = sp, E_ref = gam * m)
tw2 = twiss(ring2, rf_on = false, spin = true, at = [0])
println("recheck: perturbed ring qspin = ", replace(sprint(show, tw2.summ.sdict[:qspin]), r"\s+" => " "), "   expected -frac(a*gamma) = ", -mod(a * gam, 1))
println(failed ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
