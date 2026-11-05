"""TRK-56: Native Lie radiation ignores radiation_zero_average

Native Lie tracking ignores radiation_zero_average when applying deterministic radiation loss in a planar wiggler. The switch should subtract the loss on the zero reference orbit, which samples the same field magnitude as the on-energy orbit in this case. A 650 MeV/c electron in a five-period, 0.3 m wiggler with peak field 0.3 T loses relative momentum 2.221 × 10⁻⁸ with either switch setting; ordinary native and Runge–Kutta tracking cancel the reference loss when the switch is enabled. Removing the separate duplicated native radiation call halves the loss but still leaves the switch ineffective.

The script toggles damping and reference-loss subtraction for native Lie, standard and Runge–Kutta tracking. It checks the ordinary unshifted loss against the analytic sinusoidal-field integral and requires zero loss with damping disabled. Set BMAD_DIST to the build root; the companion one-call diagnostic isolates the duplicated-call defect.
"""

import numpy as np
from _bmad import compare, fortran, verdict
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
rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
mass, c, radius, p0, g = rows["constants"]
# Integral of curvature squared over five full periods: I2=L*G^2/2.
# Fractional deterministic loss D=(2*r_e/3)*gamma^3*I2 to first order.
if compare("wiggler curvature (1/m), B*c/p0", g, 0.3 * c / p0, atol=3e-16):
    raise RuntimeError("Field normalization control failed")
g = 0.3 * c / p0
gamma = np.hypot(p0, mass) / mass
loss = (2 * radius / 3) * gamma**3 * 0.3 * g * g / 2
print(f"analytic unshifted loss D={loss:.17g}; g_max={g:.17g} 1/m")
for method in (1, 2, 3):
    for avg in (0, 1):
        if compare(
            f"method {method}, zero-average={avg}, damping-off delta",
            rows[f"case{method}0{avg}"][5],
            0,
            atol=2e-16,
        ):
            raise RuntimeError("Damping-off control failed")
for method in (2, 3):
    if compare(
        f"method {method} unshifted delta, -D", rows[f"case{method}10"][5], -loss, atol=2e-15
    ):
        raise RuntimeError("Ordinary unshifted-loss control failed")
    if compare(
        f"method {method} shifted delta, reference cancellation",
        rows[f"case{method}11"][5],
        0,
        atol=2e-16,
    ):
        raise RuntimeError("Ordinary subtraction control failed")
print(
    f"native Lie unshifted delta={rows['case110'][5]:.17g}; shifted delta={rows['case111'][5]:.17g}"
)
print(
    f"native Lie full-record flag difference={np.max(np.abs(rows['case110'] - rows['case111'])):.17g}"
)
defect = compare(
    "native Lie shifted delta, reference cancellation", rows["case111"][5], 0, atol=2e-15
)
if defect and not np.array_equal(rows["case110"], rows["case111"]):
    raise RuntimeError("Native flag changes the returned record; inspect the subtraction model")
verdict(defect)
