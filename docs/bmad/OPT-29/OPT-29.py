"""OPT-29: Radiation uses the reference species for a tracked particle of another charge state

Tracking a particle on a reference of another charge state makes symp_lie_bmad use the reference species’ radiation coefficients. The same particle in the same magnetic field at the same momentum must have the same radiation loss, independent of the reference species. For a carbon-12 ion of charge +3, changing the reference from +3 to +6 changes the calculated loss by 0.05488%, even though the radiation-off transverse trajectory is identical. The coefficients must use the tracked particle’s classical radius and Lorentz factor; separate charge-factor and integration-step errors also affect the absolute loss.

The script tracks the same carbon-12 +3 particle through a ten-period wiggler with matched +3 and mixed +6 references and checks native reference-species invariance. Scratch copies isolate the two other radiation faults and provide a tracked-species coefficient control. BMAD_DIST supplies the build and BMAD_SOURCE supplies symp_lie_bmad.f90."""

import numpy as np
from _bmad import compare, fortran, source, verdict

PROGRAM = """program edge_probe
use bmad
use random_mod
implicit none
type(lat_struct), target :: lat
type(ele_struct), pointer :: ele
type(coord_struct) :: orb0, orb, off
real(rp) :: gamma, reference_gamma, rc, gmax, len, loss_expected, coefficient_ratio
logical err
character(400) :: lat_file
call get_command_argument(1, lat_file)
call bmad_parser(lat_file, lat, err_flag=err)
if(err) error stop 'Parsing failed'
call ran_seed_put(12345)
ele=>lat%ele(1)
call init_coord(orb0, lat%ele(0), downstream_end$)
! init_coord uses reference beta for zero delta; supply the tracked mass shell.
call convert_pc_to(orb0%p0c,orb0%species,beta=orb0%beta)
call convert_pc_to(ele%value(p0c$), orb0%species, gamma=gamma)
rc=classical_radius(orb0%species)
gmax=abs(c_light*ele%value(b_max$)*charge_of(orb0%species)/ele%value(p0c$))
len=ele%value(l$)
loss_expected=-2*rc*gamma**3*(gmax**2*len/2)/3
reference_gamma=ele%value(e_tot$)/mass_of(lat%param%particle)
coefficient_ratio=classical_radius(lat%param%particle)/rc*(reference_gamma/gamma)**3
write(*,'(a,es24.15)') 'COEFFICIENT_RATIO ',coefficient_ratio
write(*,*) 'reference ', trim(species_name(lat%param%particle)), ' tracked ', trim(species_name(orb0%species))
bmad_com%radiation_damping_on=.false.
bmad_com%radiation_fluctuations_on=.false.
call track1(orb0, ele, lat%param, off, err_flag=err)
if(err.or.off%state/=alive$) error stop 'Radiation-off track failed'
if(off%vec(6)/=0) error stop 'Radiation-off momentum changed'
write(*,'(a,6es25.16)') 'OFF ',off%vec
bmad_com%radiation_damping_on=.true.
call track1(orb0, ele, lat%param, orb, err_flag=err)
if(err.or.orb%state/=alive$) error stop 'Radiation track failed'
write(*,'(a,3es24.15)') 'loss, expected, ratio: ',orb%vec(6),loss_expected,orb%vec(6)/loss_expected
end program
"""
MATCHED = """parameter[particle] = #12C+3
parameter[default_tracking_species] = #12C+3
parameter[geometry] = open
beginning[p0c] = sqrt(1000^2-1)*mass_of(#12C+6)
w: wiggler, l = 1, n_period = 10, num_steps = 2000, tracking_method = symp_lie_bmad,
   b_max = sqrt(1000^2-1)*mass_of(#12C+6)/(c_light*charge_of(#12C+6))
ln: line = (w)
use, ln
"""
MIXED = """parameter[particle] = #12C+6
parameter[default_tracking_species] = #12C+3
parameter[geometry] = open
beginning[p0c] = sqrt(1000^2-1)*mass_of(#12C+6)
w: wiggler, l = 1, n_period = 10, num_steps = 2000, tracking_method = symp_lie_bmad,
   b_max = sqrt(1000^2-1)*mass_of(#12C+6)/(c_light*charge_of(#12C+6))
ln: line = (w)
use, ln
"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one prerequisite coefficient: {old}")
    return text.replace(old, new)


native = (source() / "bmad/low_level/symp_lie_bmad.f90").read_text()
# Isolate the reference-species defect from the separate charge-squared and
# doubled-step defects. These substitutions are compiled only in scratch.
prerequisites = replace_once(
    native,
    "2 * classical_radius_factor * gamma_0**3 * ds / (3 * mc2)",
    "2 * classical_radius(param%particle) * gamma_0**3 * ds2 / 3",
)
prerequisites = replace_once(
    prerequisites,
    "sqrt(rad_fluct_const * ds * gamma_0**5) / mc2",
    "sqrt(55 * classical_radius(param%particle) * h_bar_planck * c_light * ds2 * gamma_0**5 / (24 * sqrt_3 * mc2))",
)
tracked = replace_once(
    prerequisites, "mc2 = mass_of(param%particle)", "mc2 = mass_of(orbit%species)"
)
tracked = replace_once(
    tracked,
    "gamma_0 = ele%value(e_tot$) / mass_of(param%particle)",
    "call convert_pc_to(ele%value(p0c$), orbit%species, gamma=gamma_0)",
)
tracked = tracked.replace("classical_radius(param%particle)", "classical_radius(orbit%species)")
native_losses = []
native_orbits = []
for name, lattice in (("matched C+3 reference", MATCHED), ("mixed C+6 reference", MIXED)):
    for mode, routine in (
        ("pin coefficients", native),
        ("isolated reference-species coefficients", prerequisites),
        ("tracked-species control", tracked),
    ):
        # An external definition in this object supersedes the library member.
        log = fortran(PROGRAM + "\n" + routine, ["lat.bmad"], files={"lat.bmad": lattice}).log
        rows = [
            line.split()[-3:]
            for line in log.splitlines()
            if line.startswith("loss, expected, ratio:")
        ]
        if len(rows) != 1:
            raise RuntimeError("Missing radiation result")
        actual, expected, ratio = map(float, rows[0])
        coefficient_rows = [
            line.split()[1] for line in log.splitlines() if line.startswith("COEFFICIENT_RATIO ")
        ]
        if len(coefficient_rows) != 1:
            raise RuntimeError("Missing reference/particle coefficient ratio")
        prediction = float(coefficient_rows[0])
        print(
            f"{name}, {mode}: delta={actual:.16g}, analytic delta={expected:.16g}, ratio={ratio:.16g}"
        )
        if mode == "pin coefficients":
            if not np.isfinite([actual, expected, ratio]).all() or actual >= 0:
                raise RuntimeError("Native radiation loss must be finite and negative")
            off_rows = [
                list(map(float, line.split()[1:]))
                for line in log.splitlines()
                if line.startswith("OFF ")
            ]
            if len(off_rows) != 1 or len(off_rows[0]) != 6:
                raise RuntimeError("Missing radiation-off trajectory")
            native_losses.append(actual)
            native_orbits.append(off_rows[0][:4])
            continue
        if mode == "isolated reference-species coefficients" and compare(
            f"{name}: coefficient-ratio control", ratio, prediction, atol=2e-7
        ):
            raise AssertionError("Reference-species coefficient prediction failed")
        bad = compare(f"{name}, {mode}: loss/analytic", ratio, 1.0, atol=2e-7)
        if (name.startswith("matched") or mode == "tracked-species control") and bad:
            raise AssertionError("Matched or tracked-species control failed")
# A reference species is a coordinate convention: the same particle in the same
# field at the same physical momentum must lose the same amount of momentum.
# The verdict uses only unmodified production coefficients. Scratch interventions
# explain how this fault combines with the separate radius and step-size faults.
np.testing.assert_allclose(native_orbits[0], native_orbits[1], atol=1e-18, rtol=0)
verdict(
    compare(
        "native mixed/matched loss; reference-species invariance",
        native_losses[1] / native_losses[0],
        1.0,
        atol=2e-10,
    )
)
