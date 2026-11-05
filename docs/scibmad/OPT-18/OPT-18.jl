# OPT-18: Summary tunes and `phi` columns alias when a sampled segment exceeds half a turn
#
# `twiss` builds the summary tunes and the `phi` phase-advance columns by adding the principal phase
# angle, in (−½, ½] turn, of each interval between sampled rows, so an interval longer than half a
# turn aliases the integer part. For a ring of eight thin FODO cells, sampling only element 17
# (`at=[17]`, after four cells) gives q₁ = −0.8515 instead of 1.149 and a phase of −0.4257 instead
# of 0.5743 turns at s = 8 m; thin-lens matrix products and sampling every cell establish the
# accumulated values. The documented summary-only `at=[]` call treats the ring as one interval and
# returns 0.1485. Fractional tunes stay correct; the integer part is lost wherever an interval
# between sampled rows exceeds half a turn.
#
# The script builds a ring of eight thin FODO cells (thin quadrupoles, 1 m drifts, 1 GeV/c
# electrons, RF off) and computes the cell tune q_c from the thin-lens cell matrix; q_c is below ½,
# so its principal value is the true advance, the ring tune is 8q_c and the phase after four cells
# is 4q_c. It compares `twiss` with every element sampled (`at=:`), with only element 17 sampled
# (`at=[17]`) and with no rows (`at=[]`), and shows that the principal angle of the four-cell matrix
# is 4q_c − 1, the value the sparse run returns.
using SciBmad, LinearAlgebra
qf = Multipole(Kn1L=0.8); qd = Multipole(Kn1L=-0.7); d = Drift(L=1.0)
ring = Beamline(repeat([qf, d, qd, d], 8), species_ref=Species("electron"), pc_ref=1e9)
num(x) = x isa AmplitudeDependentValue ? getterm(x) : x
Dm = [1.0 1.0; 0 1]; K(k) = [1.0 0; -k 1]
Mc = Dm * K(-0.7) * Dm * K(0.8)
mu(m) = (c = tr(m) / 2; s = sign(m[1, 2]) * sqrt(1 - c^2); atan(s, c) / 2pi)
qc = mu(Mc)
println("analytic: cell tune = ", qc, "  ring tune 8*qc = ", 8qc, "  phase after 4 cells = ", 4qc)
println("independent: principal angle of the 4-cell matrix = ", mu(Mc^4), "  (= 4*qc - 1 = ", 4qc - 1, ")")
# Each sampled segment adds the principal angle atan(sin, cos)/2π ∈ (−½, ½] of its normalizing
# rotation, and the summary q1/q2 is the phase accumulated at END. With at=: every element is a
# segment, so the sum is the true phase advance; with sparse at, a segment longer than half a turn
# loses an integer.
tall = twiss(ring, rf_on=false, cols=["phi1"])
i8 = findfirst(==(8.0), tall.s)
println("independent: at=:     q1 = ", num(tall.q1), "  phi1 at s=8 m (entrance of element 17) = ", tall.phi1[i8])
t17 = twiss(ring, rf_on=false, at=[17], cols=["phi1"])
println("observed:    at=[17]  q1 = ", num(t17.q1), "  rows s = ", t17.s, "  phi1 = ", t17.phi1)
tnone = twiss(ring, rf_on=false, at=[])
println("observed:    at=[]    q1 = ", num(tnone.q1), "  (principal value of 8*qc = ", 8qc - 1, ")")
bad = abs(num(t17.q1) - 8qc) > 0.5 && abs(num(tall.q1) - 8qc) < 1e-12
println(bad ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
