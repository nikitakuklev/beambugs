# TRK-12: Sagan tracking guide uses nonexistent names
#
# The `SaganCavity` tracking guide's examples fail when copied, because they use the keyword
# `num_cells`, the nonexistent enum name `PhaseReference`, and `zero_phase` as a `SaganCavity`
# keyword. The constructor and export list instead provide `n_cells` and `PhaseRef`, and
# `zero_phase` belongs to the cavity element's `RFParams`. The guide's two-cell call with active
# length 1.600 m raises `MethodError`, while `SaganCavity(n_cells=2, L_active=1.6)` succeeds;
# passing `zero_phase` to that constructor also raises `MethodError`. The physics guide repeats the
# wrong cell-count keyword, so users following either guide are affected.
#
# The script calls the guide's `SaganCavity(num_cells=2, L_active=1.6)` and
# `SaganCavity(zero_phase=PhaseRef.BelowTransition)` and checks whether `PhaseReference` is defined
# in SciBmad. The control `SaganCavity(n_cells=2, L_active=1.6)` must succeed.
using SciBmad

function attempt(label, f)
  try
    f(); println(rpad(label, 46), " OK"); return false
  catch err
    println(rpad(label, 46), " ", typeof(err), ": ", first(split(sprint(showerror, err), '\n'))); return true
  end
end
# The constructor keywords are n_cells, L_active, radiation_damping_on and
# radiation_fluctuations_on; the enum is PhaseRef, and zero_phase is a field of RFParams.
e1 = attempt("SaganCavity(num_cells = 2, L_active = 1.6)", () -> SaganCavity(num_cells = 2, L_active = 1.6))
e2 = attempt("SaganCavity(zero_phase = PhaseRef.BelowTransition)", () -> SaganCavity(zero_phase = PhaseRef.BelowTransition))
e3 = !isdefined(SciBmad, :PhaseReference)
println(rpad("isdefined(SciBmad, :PhaseReference)", 46), " ", !e3)
ok = attempt("control: SaganCavity(n_cells = 2, L_active = 1.6)", () -> SaganCavity(n_cells = 2, L_active = 1.6))
println(e1 && e2 && e3 && !ok ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
