# TRK-09: RF with voltage but no frequency loses every live particle with NaN coordinates
#
# An active `RFCavity` with nonzero voltage but an omitted or zero frequency returns NaN coordinates
# and marks particles lost, with no error naming the missing input. Such a cavity must either have a
# defined zero-frequency model or be rejected as invalid input; instead the builder divides its
# phase by the zero angular frequency, which gives an undefined kick. A 1.000 m, 1.000 MV cavity
# returns NaN coordinates and `STATE_LOST` under both `SciBmadStandard` and `SaganCavity` instead of
# an error identifying the configuration mistake. Disabled RF, ignored `RFParams` and zero-voltage
# cavities remain finite and alive.
#
# The script tracks one proton through `RFCavity(L=1, voltage=1e6)` without a frequency, with
# `SciBmadStandard` and with `SaganCavity`, and reports whether tracking raised an error or returned
# NaN coordinates; an error or finite coordinates are expected. A recheck evaluates the builder's
# own helpers for this cavity: it is active, its angular frequency is 0, and the resulting phase
# time is −∞, which gives NaN when multiplied by the frequency.
using SciBmad

const sp = Species("proton"); const pc = 2 * massof(sp)
function run(ele; branch = false)
  if branch
    br = Branch([Beamline([Marker()], species_ref = sp, pc_ref = pc), Beamline([ele], dE_ref = 0.0)])
    bl = br.beamlines[2]
  else
    bl = Beamline([ele], species_ref = sp, pc_ref = pc)
  end
  b = Bunch([0.001 0.0 0.0 0.0 0.001 0.0], species = sp, p_over_q_ref = pc / C_LIGHT)
  try
    track!(b, bl; use_explicit_SIMD = false)
    return "returned v = $(vec(b.v)), state = $(Int(b.state[1]))", any(isnan, b.v)
  catch err
    return "raised: " * sprint(showerror, err), false
  end
end
# RFParams with voltage != 0 is active, so a tracking method must either model it or refuse it with
# an error; marking every particle lost reports a beam loss that the configuration did not cause.
s1, n1 = run(RFCavity(L = 1.0, voltage = 1e6))
s2, n2 = run(RFCavity(L = 1.0, voltage = 1e6, tracking_method = SaganCavity()); branch = true)
println("SciBmadStandard: ", s1)
println("SaganCavity:     ", s2)
println("expected: an error (frequency not set) or finite coordinates")
# --- Independent recheck: the builder's phase clock for this element -------------------------
let EXT = Base.get_extension(SciBmad.BeamTracking, :BeamTrackingBeamlinesExt), rp = RFCavity(L = 1.0, voltage = 1e6).RFParams
  om = EXT.rf_omega_calc(rp, nothing); tr = (EXT.rf_phi0_calc(rp, sp) - pi / 2) / om
  println("recheck: isactive = ", SciBmad.Beamlines.isactive(rp), ", rate_meaning = ", rp.rate_meaning,
          ", omega = ", om, ", t_ref = (phi - pi/2)/omega = ", tr, ", omega*t_ref = ", om * tr)
end
println(n1 && n2 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
