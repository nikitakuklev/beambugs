# COL-03: SaganCavity fluctuations with multipoles throw `MethodError`
#
# Tracking a finite-length `SaganCavity` that carries magnetic multipoles with
# `radiation_fluctuations_on` raises `MethodError` instead of applying the random radiation kicks.
# The cavity's radiation helper calls `stochastic_radiation!` without the longitudinal-position and
# backend arguments that its multipole method requires. A 1.000 m two-cell cavity with normalized
# quadrupole strength `Kn1` = 0.05000 m⁻² tracks with fluctuations off and fails with them on, while
# the same call executes once the two missing arguments and a nonzero kick length are supplied.
# Source inspection extends the failure to cavities with a solenoid field and shows that pure-RF and
# zero-length cavities never reach this helper.
#
# The script tracks one electron through a 1 m `RFCavity` with `Kn1` = 0.05 m⁻² using
# `SaganCavity(n_cells=2)` with radiation damping on, first with fluctuations off (a control that
# must track) and then with them on, and reports the `MethodError`. It takes the failing call's
# arguments from the error, checks that no method accepts them, inserts the missing position `s` and
# backend after the coordinates, and executes that call with a fixed random seed and a 0.25 m
# length. Because the arguments come from the error, the check stays valid if the helper's argument
# list changes.
using SciBmad, Random
const BT = SciBmad.BeamTracking

sp = Species("electron"); E0 = 1e9
for fl in (false, true)
    m = Marker(E_ref = E0, species_ref = sp)
    c = RFCavity(L = 1.0, voltage = 1e6, rf_frequency = 5e8, dE_ref = 0.0, Kn1 = 0.05,
                 tracking_method = SaganCavity(n_cells = 2, radiation_damping_on = true,
                                               radiation_fluctuations_on = fl))
    br = Branch([m, c])                      # same construction as BeamTracking's Sagan tests
    cav = br.beamlines[2].line[1]
    b = Bunch(v = [1e-3 0.0 1e-3 0.0 0.0 0.0], species = sp, p_over_q_ref = BT.E_to_R(sp, E0))
    try
        track!(b, cav)
        println("fluctuations=", fl, ": tracked, v = ", vec(b.coords.v))
    catch err
        msg = first(split(sprint(showerror, err), '\n'))
        println("fluctuations=", fl, ": ", typeof(err), ": ", msg[1:min(end, 70)], " ...")
        st = stacktrace(catch_backtrace())
        k = findfirst(f -> occursin("sagan_cavity.jl", string(f.file)), st)
        println("  thrown from ", st[k].func, " at ", basename(string(st[k].file)), ":", st[k].line)
        global failed = err
    end
end
if @isdefined(failed) && failed isa MethodError
    # The helper calls stochastic_radiation! without the `s` and `backend` arguments that the only
    # multipole method takes after the coordinates: 11 arguments against 13 before the radiation
    # rotation arguments were added, 12 against 14 after.
    call = failed.args                       # arguments of the failing call at sagan_cavity.jl:302
    println("applicable(stochastic_radiation!, ", length(call), "-argument call used by sagan_cavity.jl:302) = ",
            applicable(BT.stochastic_radiation!, call...))
    # Independent recheck: the same helper arguments with s and backend inserted after the coordinates. The
    # failing call passes a zero length for this cavity, which would make the kick vanish, so the control
    # uses a 0.25 m length.
    c = Bunch(v = [1e-3 0.0 1e-3 0.0 0.0 0.0]).coords
    args = (call[1], c, 0.0, BT.CPU(), call[3:end-1]..., 0.25)
    println("recheck: applicable with (s, backend) inserted = ", applicable(BT.stochastic_radiation!, args...))
    Random.seed!(1)
    BT.stochastic_radiation!(args...); println("recheck: ", length(args), "-argument call executes, dpz = ", c.v[1, 6])
end
println(@isdefined(failed) && failed isa MethodError ? "DEFECT REPRODUCED" : "NOT REPRODUCED")
