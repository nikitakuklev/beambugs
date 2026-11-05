# LAT-08: `track` rejects the documented vector `weight`
#
# Passing per-particle weights to `track` as the documented vector of length `n_particles` raises a
# `TypeError` before tracking begins, because the keyword is declared
# `Union{AbstractMatrix,Nothing}`. `track(...; weight=[0.25, 0.75])` therefore fails on documented
# input. The same vector passed through an explicit `Bunch` is kept as [0.25, 0.75] in a zero-turn
# control, confirming that the underlying bunch API accepts it.
#
# The script tracks two electrons through a 1 m drift for zero turns, passing `weight=[0.25, 0.75]`
# to `track`, and as a control passes the same weights inside an explicit `Bunch`. The resulting
# bunch must carry the weights [0.25, 0.75].
using SciBmad
bl = Beamline([Drift(L=1.0)], species_ref=Species("electron"), E_ref=1e9)
v0 = [1e-3 1e-4 0 0 0 0; 2e-3 0 0 0 0 0]
w = [0.25, 0.75]
r = try
    res = track(bl; v0=v0, weight=w, n_turns=0)
    res.bunch.weight
catch e
    "ERROR: " * first(split(sprint(showerror, e), '\n'))
end
println("observed track(...; weight=[0.25,0.75]) -> bunch.weight = ", r)
b = Bunch(v=copy(v0), weight=copy(w), species=bl.species_ref, p_over_q_ref=bl.p_over_q_ref)
res2 = track(bl; bunch=b, n_turns=0)
println("control  track(...; bunch=Bunch(..., weight=[0.25,0.75])) -> bunch.weight = ", res2.bunch.weight)
println("expected bunch.weight = ", w)
println(r == w ? "NOT REPRODUCED" : "DEFECT REPRODUCED")
