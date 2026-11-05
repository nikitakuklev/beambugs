# TRK-06: `SaganCavity` leaves `Bunch.p_over_q_ref` stale; the next element erases the energy gain
#
# An accelerating `SaganCavity` renormalizes particle coordinates to the exit reference momentum but
# leaves the bunch's `p_over_q_ref` at its entrance value, so the next element repeats the reference
# shift and erases the energy gain. The bunch normalization contract requires the synchronous
# particle to keep the gain at zero relative momentum offset δ, and ordinary RF tracking with the
# same reference-energy change does so. A proton with reference momentum p₀c = 2mc², twice its rest
# energy, crossing a zero-length 1.000 MV cavity and then a 1.000 m drift gains 0 instead of 1.000
# MeV, and its δ becomes −5.954 × 10⁻⁴ instead of 0.
#
# The script builds the documented `Branch` partition: a proton at p₀c = 2mc², then a zero-length 1
# MV `SaganCavity` and a 1 m drift with reference energy gain `dE_ref` = 1 MeV. It tracks the
# reference particle element by element and through the whole beamline, and prints δ, the bunch
# `p_over_q_ref` and the energy implied by the bunch metadata, which must have risen by 1 MeV. A
# control repeats this with ordinary RF (at `phi0 = π`, which compensates the ordinary crest phase
# error), which keeps the gain.
using SciBmad

const sp = Species("proton"); const pc = 2 * massof(sp); const M = massof(sp)
const E0 = hypot(pc, M)
energy(b) = hypot(b.p_over_q_ref * C_LIGHT * chargeof(sp) * (1 + b.v[1, 6]), M)

br = Branch([Beamline([Marker()], species_ref = sp, pc_ref = pc),
             Beamline([RFCavity(L = 0.0, voltage = 1e6, rf_frequency = 1e6, tracking_method = SaganCavity()),
                       Drift(L = 1.0)], dE_ref = 1e6)])
m, acc = br.beamlines
b = Bunch(zeros(1, 6), species = sp, p_over_q_ref = pc / C_LIGHT)
track!(b, m; use_explicit_SIMD = false)
track!(b, acc.line[1]; use_explicit_SIMD = false)
# Bunch.p_over_q_ref defines the coordinate normalization: an on-crest gain equal to dE_ref must
# leave the reference particle at delta = 0 with energy E0 + 1 MeV after any later element.
println("after Sagan cavity: delta = ", b.v[1, 6], "  bunch R = ", b.p_over_q_ref, "  beamline R = ", acc.p_over_q_ref)
println("                    energy from bunch metadata - E0 = ", energy(b) - E0, " eV (expected 1e6)")
track!(b, acc.line[2]; use_explicit_SIMD = false)
println("after 1 m drift:    delta = ", b.v[1, 6], "  bunch R = ", b.p_over_q_ref)
println("                    energy from bunch metadata - E0 = ", energy(b) - E0, " eV (expected 1e6)")
b2 = Bunch(zeros(1, 6), species = sp, p_over_q_ref = pc / C_LIGHT)
track!(b2, m; use_explicit_SIMD = false); track!(b2, acc; use_explicit_SIMD = false)
println("whole-beamline track!: energy - E0 = ", energy(b2) - E0, " eV (expected 1e6)")
# --- Independent recheck: the ordinary RF path with the same dE_ref keeps the gain -----------
br2 = Branch([Beamline([Marker()], species_ref = sp, pc_ref = pc),
              Beamline([RFCavity(L = 1e-6, voltage = 1e6, rf_frequency = 1e6, phi0 = pi), Drift(L = 1.0)], dE_ref = 1e6)])
b3 = Bunch(zeros(1, 6), species = sp, p_over_q_ref = pc / C_LIGHT)
track!(b3, br2.beamlines[1]; use_explicit_SIMD = false); track!(b3, br2.beamlines[2]; use_explicit_SIMD = false)
println("recheck: ordinary RF (phi0 = pi, see TRK-01) + drift: delta = ", b3.v[1, 6], "  bunch R = ", b3.p_over_q_ref,
        "  energy - E0 = ", energy(b3) - E0, " eV")
println(abs(energy(b) - E0) < 1e-3 && abs(energy(b2) - E0) < 1e-3 ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
