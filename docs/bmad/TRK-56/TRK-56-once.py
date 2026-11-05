"""TRK-56: One-call radiation diagnostic

This diagnostic removes one duplicated radiation call per body step from a scratch copy of symp_lie_bmad and reruns the reference-loss switch cases. It checks the unshifted loss against the analytic sinusoidal-field integral and validates ordinary-map reference cancellation. Set BMAD_DIST and BMAD_SOURCE to matching build and source roots.
"""

import numpy as np
from _bmad import compare, fortran, source, verdict
from _orbital_matrices_radiation import records

PROGRAM = """
program probe
use bmad

implicit none
type(lat_struct), target :: lat
type(coord_struct) a,b
real(rp) v(6)
logical err
integer method,damp,avg
character(12) label
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
call ran_seed_put(1729)
bmad_com%radiation_fluctuations_on=.false.
write(*,'(a,5es26.17)') 'RESULT constants ',m_electron,c_light,classical_radius(electron$), &
 lat%ele(1)%value(p0c$),lat%ele(1)%value(g_max$)
do method=1,3
 if(method==1) lat%ele(1)%tracking_method=symp_lie_bmad$
 if(method==2) lat%ele(1)%tracking_method=bmad_standard$
 if(method==3) lat%ele(1)%tracking_method=runge_kutta$
 do damp=0,1
  do avg=0,1
   bmad_com%radiation_damping_on=(damp==1)
   bmad_com%radiation_zero_average=(avg==1)
   v=0
   call init_coord(a,v,lat%ele(1),upstream_end$,particle=electron$)
   call track1(a,lat%ele(1),lat%param,b,err_flag=err)
   if(err.or.b%state/=alive$) error stop 'radiation tracking failed'
   write(*,'(a,3i1,8es26.17)') 'RESULT case',method,damp,avg,b%vec,b%beta,b%t
  enddo
 enddo
enddo
end program
"""
LATTICE = """
no_digested
parameter[geometry]=open
parameter[particle]=electron
beginning[p0c]=650e6
e: wiggler,l=.3,l_period=.06,b_max=.3,field_calc=planar_model,kx=0,tracking_method=symp_lie_bmad,ds_step=.0005
lat: line=(e)
use,lat
"""
routine = (source() / "bmad/low_level/symp_lie_bmad.f90").read_text()
fragment = "    call radiation_kick(orbit)\n    if (bmad_com%spin_tracking_on) call wig_spin(orbit, s_pos, ds2)\n    call radiation_kick(orbit)"
if routine.count(fragment) != 2:
    raise RuntimeError("Expected the two duplicated-call sites in the matching source")
routine = routine.replace(fragment, fragment.split("\n", 1)[1])
rows = records(fortran(routine + "\n" + PROGRAM, files={"lat.bmad": LATTICE}).log)
mass, c, radius, p0, g = rows["constants"]
if compare("one-call curvature (1/m), B*c/p0", g, 0.3 * c / p0, atol=3e-16):
    raise RuntimeError("Field normalization control failed")
# I2=L*(B*c/p0)^2/2 over complete sinusoidal periods; D=(2*r_e/3)*gamma^3*I2.
loss = (2 * radius / 3) * (np.hypot(p0, mass) / mass) ** 3 * 0.3 * (0.3 * c / p0) ** 2 / 2
for method in (1, 2, 3):
    for avg in (0, 1):
        if compare(
            f"one-call method {method}, zero-average={avg}, damping-off delta",
            rows[f"case{method}0{avg}"][5],
            0,
            atol=2e-16,
        ):
            raise RuntimeError("Damping-off control failed")
for method in (1, 2, 3):
    if compare(
        f"one-call method {method} unshifted delta, analytic -D",
        rows[f"case{method}10"][5],
        -loss,
        atol=2e-15,
    ):
        raise RuntimeError("One-call loss-scale control failed")
for method in (2, 3):
    if compare(
        f"one-call ordinary method {method} shifted delta, reference cancellation",
        rows[f"case{method}11"][5],
        0,
        atol=2e-16,
    ):
        raise RuntimeError("Ordinary subtraction control failed")
defect = compare(
    "one-call native shifted delta, reference cancellation", rows["case111"][5], 0, atol=2e-15
)
if defect and not np.array_equal(rows["case110"], rows["case111"]):
    raise RuntimeError("Native flag changes the returned record; inspect the subtraction model")
print(
    f"one-call native unshifted/shifted delta: {rows['case110'][5]:.17g}, {rows['case111'][5]:.17g}"
)
verdict(defect)
