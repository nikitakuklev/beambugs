# OPT-10: Experimental torus and aperture code calls APIs that do not exist
#
# Several entry points of the experimental torus tools in `SciBmad.Experimental` fail. On a ring
# without RF, the default tune guess of `max_one_torus` and `walk_one_torus` reads the synchrotron
# tune `q3`, which `twiss` provides only with RF, and errors with `Twiss does not have q3`; passing
# `Q_guess` explicitly avoids this failure.
# `Experimental.dynamic_aperture(delta_dependent_orbits=false)` calls `walk_J3`, which is defined
# nowhere. By source analysis, the `ais` output of `walk_one_torus` cannot be produced: one branch
# is unreachable and the other calls the undefined `freq_res_other!`. Introspection of the loaded
# package confirms that the summary key and both functions are missing.
#
# The script builds a periodic cell of thin quadrupoles and 1 m drifts for 1 GeV/c electrons without
# RF. It calls `max_one_torus` with the default tune guess and `Experimental.dynamic_aperture` with
# `delta_dependent_orbits=false`, prints the first line of each error, and reads `tw.table` as the
# unreachable `ais` branch would. It then inspects the loaded package: the `twiss` summary keys,
# whether `walk_J3` and `freq_res_other!` are defined, and whether an n × 2 view compares equal to
# 0, the test that makes the first `ais` branch unreachable.
using SciBmad
const EX = SciBmad.Experimental
qf = Multipole(Kn1L = 0.8); qd = Multipole(Kn1L = -0.7); d = Drift(L = 1.0)
bl = Beamline([qf, d, qd, d], species_ref = Species("electron"), pc_ref = 1e9)
first_line(f) = try f(); "no error" catch err; first(split(sprint(showerror, err), '\n')) end

# (a) The default Q_guess evaluates [q1, q2, q3] of twiss(bl; at=[]); a coasting summary has no q3,
# and Twiss errors on names it does not hold.
ra = first_line(() -> EX.max_one_torus(bl, ones(Int, 1, 1), ones(Int, 1, 1), fill(1e-4, 1, 1), 1; verbose = false))
println("(a) max_one_torus default Q_guess -> ", ra)
# (b) With delta_dependent_orbits=false, Experimental.dynamic_aperture calls walk_J3, which is
# defined nowhere in SciBmad.
rb = first_line(() -> EX.dynamic_aperture(bl; n_r = 2, n_theta = 2, deltas = [0.0], nsig_1 = 1, nsig_2 = 1,
                                          emit_1 = 1e-9, emit_2 = 1e-9, n_turns = 1, delta_dependent_orbits = false, verbose = false))
println("(b) Experimental.dynamic_aperture(delta_dependent_orbits=false) -> ", rb)
# (c) The ais branch of walk_one_torus tests coord == 0 on an n_particles × 2 view (always false)
# and would read tw.table; the other branch differentiates the undefined freq_res_other!.
tw = twiss(bl, rf_on = false)
println("(c) tw.table -> ", first_line(() -> tw.table))
cv = view(zeros(1, 2, 1), :, :, 1) == 0
fo = isdefined(EX, :freq_res_other!)
println("independent: summary keys = ", keys(tw.summ.sdict),
        ", isdefined(Experimental, :walk_J3) = ", isdefined(EX, :walk_J3), ", isdefined(SciBmad, :walk_J3) = ", isdefined(SciBmad, :walk_J3),
        ", isdefined(Experimental, :freq_res_other!) = ", fo, ", (coord view == 0) = ", cv)
ok = occursin("q3", ra) && occursin("walk_J3", rb) && !fo && !cv
println(ok ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
