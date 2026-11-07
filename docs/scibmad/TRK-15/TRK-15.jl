# TRK-15: `SciBmadStandard` docstring names a nonexistent `Yoshida` integrator
#
# The docstring of the default tracking method `SciBmadStandard` says that elements without an exact
# map use "the symplectic integrator `Yoshida(order=4, num_steps=1)`". No `Yoshida` type exists in
# SciBmad, Beamlines or BeamTracking, and no tracking method has a `num_steps` field. The default
# method maps to `Symplectic` with keyword `n_steps`, at order 2 for exact bodies and order 4
# otherwise. A user who copies the documented method gets an `UndefVarError`; tracking itself is
# unaffected.
#
# The script reads the `SciBmadStandard` docstring of the loaded Beamlines and checks whether
# `Yoshida` is defined in SciBmad, Beamlines or BeamTracking. It reports the defect while the
# docstring names `Yoshida` and no such name is defined, so its verdict reflects the build it runs on.
using SciBmad
const BL = SciBmad.Beamlines
const BT = SciBmad.BeamTracking

doc = replace(string(@doc BL.SciBmadStandard), r"\s+" => " ")
doc_yoshida = occursin("Yoshida(order=4, num_steps=1)", doc)
println("SciBmadStandard docstring names Yoshida(order=4, num_steps=1): ", doc_yoshida)
defined = isdefined(SciBmad, :Yoshida) || isdefined(BL, :Yoshida) || isdefined(BT, :Yoshida)
println("Yoshida defined in SciBmad, Beamlines or BeamTracking: ", defined)
# SciBmadStandard maps to Symplectic with keyword n_steps (order 2 for exact bodies, 4 otherwise).
println("Symplectic fields: ", fieldnames(BT.Symplectic))
println(doc_yoshida && !defined ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
