# TRK-10: Methods other than `RungeKutta` silently ignore `EMFieldParams`
#
# Adding `EMFieldParams` to an element while keeping the default tracking method silently omits the
# supplied electromagnetic field. The field-definition contract and an independent Lorentz-impulse
# calculation require its force, which `RungeKutta` applies correctly. A proton with reference
# momentum p₀c = 2mc², twice its rest energy, crossing 1.000 m of uniform vertical field 0.01000 T
# exits with normalized horizontal momentum 0 instead of −1.598 × 10⁻³. In tests `SciBmadStandard`,
# `Symplectic`, `DriftKick` and `Exact` omit magnetic fields and `SaganCavity` omits a superposed
# electric field. Source inspection also finds unchecked `EMultipoleParams` on the map, patch and
# four-potential branches.
#
# The script tracks a proton with p₀c = 2mc² through a 1 m `LineElement` whose `em_field` is a
# uniform vertical field of 0.01 T, with `SciBmadStandard` and with `RungeKutta(n_steps=50)`. It
# compares the exit horizontal momentum with the impulse −B_y·L/R, where R is the magnetic rigidity,
# and would also accept an error; `RungeKutta` is the control that applies the field.
using SciBmad

const sp = Species("proton"); const pc = 2 * massof(sp); const R = pc / C_LIGHT
field(x, y, s, t, p = nothing) = (0.0, 0.0, 0.0, 0.0, 0.01, 0.0)   # uniform By = 0.01 T
function out(tm)
  ele = LineElement(L = 1.0, em_field = field, tracking_method = tm)
  b = Bunch(zeros(1, 6), species = sp, p_over_q_ref = R)
  try
    track!(b, Beamline([ele], species_ref = sp, pc_ref = pc); use_explicit_SIMD = false)
    return vec(b.v)
  catch err
    return sprint(showerror, err)
  end
end
# EMFieldParams defines the element's field and is an active group; other unsupported active groups
# are rejected with an error, so the field must deflect the beam by px = -By L/R or raise an error.
o_std = out(SciBmadStandard()); o_rk = out(RungeKutta(n_steps = 50))
println("SciBmadStandard: ", o_std)
println("RungeKutta:      ", o_rk)
println("expected px ~ -By L / R = ", -0.01 / R, " (or an error)")
bad = o_std isa Vector && all(iszero, o_std) && o_rk isa Vector && abs(o_rk[2] + 0.01 / R) < 1e-6
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
