# TRK-02: `SaganCavity` swaps the below/above-transition zero phases
#
# With `SaganCavity` tracking, the `BelowTransition` and `AboveTransition` zero-phase references
# select each other's stable zero crossing, so the longitudinal restoring kick has the wrong sign
# and the cavity runs on the unstable slope. Below transition a late particle must gain energy and
# above transition it must lose energy, as the phase-stability condition and independent tracking in
# a drift-only ring establish. In a 1.000 kV, 100.0 MHz cavity, a proton or electron at 0.8944 times
# light speed arriving late, at longitudinal offset z = −0.02000 m, loses 46.85 eV below transition
# instead of gaining 46.85 eV. The `Accelerating` crest and ordinary RF transition phases are
# correct.
#
# The script tracks a proton and an electron with p₀c = 2mc², late by z = −0.02 m, through a 1 kV,
# 100 MHz cavity at `phi0 = 0` under both transition references, with `SaganCavity` and with
# ordinary RF as the control. It compares each energy change with +|q|V·sin(ωΔt) below transition
# and −|q|V·sin(ωΔt) above, where Δt is the particle's lateness and ω the angular RF frequency. A
# recheck follows a proton for 40 turns in a drift-only ring, which is below transition, and prints
# the growth of its longitudinal amplitude: a correct `BelowTransition` cavity keeps it bounded and
# a correct `AboveTransition` one does not.
using SciBmad

function gain(species, method, zp, z)
  M = massof(species); pc = 2M; E0 = hypot(pc, M)
  ele = RFCavity(L = (method isa SaganCavity ? 0.0 : 1e-6), voltage = 1000.0, rf_frequency = 1e8,
                 phi0 = 0.0, zero_phase = zp, tracking_method = method)
  if method isa SaganCavity
    br = Branch([Beamline([Marker()], species_ref = species, pc_ref = pc), Beamline([ele], dE_ref = 0.0)])
    bl = br.beamlines[2]
  else
    bl = Beamline([ele], species_ref = species, pc_ref = pc)
  end
  v = zeros(1, 6); v[1, 5] = z
  b = Bunch(v, species = species, p_over_q_ref = pc / (C_LIGHT * chargeof(species)))
  track!(b, bl; use_explicit_SIMD = false)
  return hypot(pc * (1 + b.v[1, 6]), M) - E0
end

bad = false
for s in (Species("proton"), Species("electron"))
  M = massof(s); beta = 2 / sqrt(5)              # pc = 2M
  z = -0.02                                      # late particle
  dt = -z / (beta * C_LIGHT); w = 2pi * 1e8
  # z = beta*c*(t_ref - t), so z < 0 is late. phi0 = 0 is the stable zero crossing: below transition
  # a late particle gains |q|V sin(w dt), above transition it loses it.
  for (zp, sgn) in ((PhaseRef.BelowTransition, +1), (PhaseRef.AboveTransition, -1))
    exp_dE = sgn * 1000 * sin(w * dt)
    dS = gain(s, SaganCavity(n_cells = 0, L_active = 0.0), zp, z)
    dO = gain(s, SciBmadStandard(), zp, z)
    println(rpad(nameof(s), 9), rpad(string(zp), 32), " Sagan dE = ", dS, "  ordinary dE = ", dO,
            "  expected = ", exp_dE)
    global bad |= abs(dS + exp_dE) < 1e-4 * abs(exp_dE) && abs(dO - exp_dE) < 1e-4 * abs(exp_dE)
  end
end
# --- Independent recheck: multi-turn synchrotron stability in a drift-only ring -------------
# A drift-only ring has momentum compaction 0, i.e. it is below transition (slip -1/gamma^2).
# A correct BelowTransition cavity is stable there; a correct AboveTransition one is unstable.
function ring_growth(method, zp)
  M = massof(Species("proton")); pc = M; s0 = Species("proton")
  cav = RFCavity(L = (method isa SaganCavity ? 0.0 : 1e-6), voltage = 4.5e7, rf_frequency = 1e7,
                 zero_phase = zp, tracking_method = method)
  br = Branch([Beamline([Marker()], species_ref = s0, pc_ref = pc), Beamline([cav, Drift(L = 10.0)], dE_ref = 0.0)])
  b = Bunch([0.0 0.0 0.0 0.0 1e-3 0.0], species = s0, p_over_q_ref = pc / C_LIGHT)
  track!(b, br.beamlines[1]; use_explicit_SIMD = false)
  zmax = 0.0
  for _ in 1:40
    track!(b, br.beamlines[2]; use_explicit_SIMD = false); zmax = max(zmax, abs(b.v[1, 5]))
  end
  return zmax / 1e-3
end
for (lab, m) in (("ordinary", SciBmadStandard()), ("Sagan", SaganCavity(n_cells = 0, L_active = 0.0))),
    zp in (PhaseRef.BelowTransition, PhaseRef.AboveTransition)
  println("recheck: drift ring, ", rpad(lab, 9), rpad(string(zp), 28), " max|z|/z0 over 40 turns = ", ring_growth(m, zp))
end
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
