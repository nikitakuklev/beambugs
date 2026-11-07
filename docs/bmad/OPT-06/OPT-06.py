"""OPT-06: Radiation convergence tests multiply errors by tolerances

Tightening radiation-integration tolerances makes the convergence test easier to satisfy because it multiplies estimated errors by the tolerances. For a 3.2 m, 1 T planar wiggler at 3 GeV with 4 or 12 periods, the curvature-squared integral I₂ is 0.008871 m⁻¹ instead of 0.01598 m⁻¹, an underestimate of 44.48%. A 13-period case is 0.03527% low, and changing the tolerance from 10⁻⁴ to 10⁻⁸ leaves all three results unchanged. Dividing by the allowed errors in a temporary source copy recovers the analytic period average at 10⁻⁸.

The script runs 4-, 12- and 13-period wigglers at two tolerances and checks the curvature-squared integral against the analytic period average. It then compiles an isolated copy of rad_6d_mod with only the convergence multiplications changed to divisions and repeats the cases.
"""

import math

from _bmad import compare, fortran, source, verdict
from _s3_14_optics_matrix_radiation import control, rows

PROGRAM = """
program wigprobe
use bmad
use rad_6d_mod
implicit none
type(lat_struct),target :: lat
type(coord_struct) o1,o2
type(rad_map_struct) rm
type(rad_int1_struct) ri
type(normal_modes_struct) mode
type(coord_struct),allocatable :: orb(:)
type(ele_struct),pointer :: ele
real(rp) tol
integer i,icache,stat
logical err
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
ele=>lat%ele(1)
call init_coord(o1, [0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp,0.0_rp],ele,upstream_end$)
call make_mat6(ele,lat%param,o1,o2,err)
if (err) error stop 'matrix failed'
call row('constants',[ele%value(l$),ele%value(g_max$)])
call twiss_and_track(lat,orb,stat)
if (stat /= ok$) error stop 'twiss failed'
icache=-2
call radiation_integrals(lat,orb,mode,icache)
call row('conventional',mode%synch_int(2:3))
do i=1,2
  tol=10.0_rp**(-4*i)
  call rad1_damp_and_stoc_mats(ele,.true.,o1,o2,rm,tol,tol,err,lat%ele(0),ri)
  if (err) error stop 'radiation failed'
  if (i == 1) call row('loose',[tol,ri%i2])
  if (i == 2) call row('tight',[tol,ri%i2])
enddo
contains
subroutine row(name,values)
character(*) name
real(rp) values(:)
write(*,'(a,1x,a,100(1x,es25.16))') 'RESULT',name,values
end subroutine
end program
"""
LATTICE = "no_digested\nparameter[geometry] = open\nparameter[e_tot] = 3e9\nparameter[no_end_marker] = T\nbeginning[beta_a] = 10\nbeginning[beta_b] = 12\nw: wiggler, l = 3.2, n_period = {periods}, b_max = 1, field_calc = planar_model, tracking_method = symp_lie_bmad, num_steps = 512\nln: line = (w)\nuse, ln\n"

module = (source() / "bmad/modules/rad_6d_mod.f90").read_text()
old = "d_max = max(eps_damp*maxval(abs(ddamp)), eps_damp*maxval(abs(ddvec)), eps_stoc*maxval(abs(dstoc)))"
new = "d_max = max(maxval(abs(ddamp))/eps_damp, maxval(abs(ddvec))/eps_damp, maxval(abs(dstoc))/eps_stoc)"
if module.count(old) != 1:
    raise RuntimeError("Expected exactly one convergence expression in BMAD_SOURCE")
normalized = module.replace(old, new)
# Magnetic rigidity gives gmax = c*B/(p0*c) with B in tesla and p0*c in eV.
# The field, energy and length are fixture inputs; mc^2 is the electron rest energy.
length, energy, mass, field = 3.2, 3e9, 510998.95069, 1.0
g = 299792458.0 * field / math.sqrt(energy**2 - mass**2)
# Complete cosine periods give the independent integral L*gmax^2/2.
expected = length * g * g / 2
defect = False
for periods in (4, 12, 13):
    lattice = LATTICE.format(periods=periods)
    r = rows(fortran(PROGRAM, files={"lat.bmad": lattice}).log)
    control(
        f"{periods} periods: length (m), peak curvature (1/m) from magnetic rigidity",
        r["constants"],
        [length, g],
        atol=2e-16,
        rtol=2e-15,
    )
    # The conventional integrator provides a separate passing control.
    control(
        f"{periods} periods: conventional I2 (1/m)",
        r["conventional"][0],
        expected,
        atol=2e-12,
    )
    for key in ("loose", "tight"):
        tol, integral = r[key]
        defect |= compare(
            f"{periods} periods, tol={tol:g}: I2 (1/m), L*gmax^2/2", integral, expected, atol=2e-12
        )
        print(f"{periods} periods, tol={tol:g}: relative I2 error={integral / expected - 1:.16g}")
    corrected = rows(fortran(normalized + "\n" + PROGRAM, files={"lat.bmad": lattice}).log)
    control(
        f"{periods} periods: normalized-error diagnostic, tol=1e-8",
        corrected["tight"][1],
        expected,
        atol=2e-12,
    )
    print(
        f"{periods} periods: normalized-error diagnostic, tol=1e-4 I2={corrected['loose'][1]:.16g}"
    )
verdict(defect)
