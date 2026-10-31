# LAT-03: `R_to_v` and `R_to_gamma` omit the charge state; ion harmonic RF frequency is superluminal
#
# Beamlines' `R_to_v`, which converts a reference rigidity R to a speed, mishandles the charge state
# q: it computes |q|·c/√(1 + (m/Rc)²) instead of c/√(1 + (m/(qRc))²) = c·pc/E. A harmonic-number
# cavity's `rf_frequency` getter uses this speed, so it is wrong for multiply charged ions; the
# correct speed is consistent with the separately checked drift reference time. For ⁴He²⁺ at a
# reference momentum of 10 GeV/c in a 10 m line with `harmon` = 1, the helper returns 4.807×10⁸ m/s,
# above c, instead of 2.809×10⁸ m/s, and the getter returns 48.07 MHz instead of 28.09 MHz. Singly
# charged particles and the drift reference clock are unaffected.
#
# The script builds a 10 m `RFCavity` Beamline with `harmon` = 1 for ⁴He²⁺ at `pc_ref` = 10 GeV/c.
# It compares the speed from Beamlines' `R_to_v`, the `rf_frequency` getter and the RF frequency of
# the tracking setup with v = c·pc/E and h·v/L, and BeamTracking's `R_to_gamma` with E/m.
# BeamTracking's own `R_to_v` is printed alongside.
using SciBmad
const BT = SciBmad.BeamTracking
ext = Base.get_extension(BT, :BeamTrackingBeamlinesExt)
he = Species("#4He++")
q = chargeof(he); m = massof(he)
pc = 10e9; L = 10.0
bl = Beamline([RFCavity(L=L, harmon=1, voltage=0.0)], species_ref=he, pc_ref=pc)
R = bl.p_over_q_ref
# With pc = q R c, v = c pc/sqrt(pc^2 + m^2) = c/sqrt(1 + (m/(q R c))^2) and gamma = sqrt((q R c/m)^2 + 1);
# the helpers compute |q| c/sqrt(1 + (m/(R c))^2) and sqrt((R c/m)^2 + 1), correct only for |q| = 1.
v_exp = C_LIGHT * pc / sqrt(pc^2 + m^2)
println("species charge = ", q, ", R = ", R, " T m, pc_ref = ", bl.pc_ref, ", E_ref = ", bl.E_ref)
println("Beamlines.R_to_v   = ", Beamlines.R_to_v(he, R), " m/s")
println("BeamTracking.R_to_v= ", BT.R_to_v(he, R), " m/s")
println("expected v = c*pc/sqrt(pc^2+m^2) = ", v_exp, " m/s (c = ", C_LIGHT, ")")
f_obs = bl.line[1].rf_frequency
w_obs = ext.rf_omega_calc(bl.line[1].RFParams, bl.line[1].BeamlineParams)
println("rf_frequency getter = ", f_obs, " Hz; tracking omega/2pi = ", w_obs / (2pi), " Hz; expected h*v/L = ", v_exp / L, " Hz")
g_obs = BT.R_to_gamma(he, R); g_exp = bl.E_ref / m
println("BeamTracking.R_to_gamma = ", g_obs, "; expected E/m = ", g_exp)
bad = !isapprox(f_obs, v_exp / L; rtol=1e-14) || !isapprox(w_obs / (2pi), v_exp / L; rtol=1e-14) ||
      !isapprox(g_obs, g_exp; rtol=1e-14)
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
