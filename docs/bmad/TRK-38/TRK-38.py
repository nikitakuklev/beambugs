"""TRK-38: Radiation kicks leave the stored particle beta stale

A deterministic radiation kick changes the momentum coordinate without updating the particle record’s speed β = v/c. The record should satisfy β = p/√(p² + m²c²) at the new physical momentum. A half-element kick in a 10 m bend of curvature 0.1 m⁻¹ at 10 GeV/c changes relative momentum by −7.040 × 10⁻⁴ but leaves β too high by 1.840 × 10⁻¹². The effect is small at this energy but exceeds floating-point rounding.

The script applies the entrance radiation kick with damping disabled and enabled, keeping fluctuations off. It reconstructs β from the returned momentum and compiled electron mass and checks the damping-off control. Set BMAD_DIST to the build root.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _orbital_matrices_radiation import records

PROGRAM = """
program probe
use bmad
use radiation_mod, only: track1_radiation
implicit none
type(lat_struct), target :: lat
type(coord_struct) a,b
real(rp) v(6)
logical err
integer j
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
v=0
bmad_com%radiation_fluctuations_on=.false.
bmad_com%radiation_zero_average=.false.
do j=0,1
 bmad_com%radiation_damping_on=(j==1)
 call init_coord(a,v,lat%ele(1),upstream_end$)
 call track1_radiation(a,lat%ele(1),start_edge$)
 if(a%state/=alive$) error stop 'radiation lost particle'
 write(*,'(a,i0,4es26.17)') 'RESULT case',j,a%vec(6),a%beta,a%p0c,m_electron
enddo
end program
"""
LATTICE = """
no_digested
parameter[geometry]=open
parameter[particle]=electron
beginning[p0c]=1e10
e: sbend,l=10,g=.1,fringe_type=none
lat: line=(e)
use,lat
"""
rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)


def beta(row):
    delta, _, p0, mass = row
    p = p0 * (1 + delta)
    return p / np.hypot(p, mass)


if compare(
    "damping-off beta, relativistic momentum relation",
    rows["case0"][1],
    beta(rows["case0"]),
    atol=3e-16,
):
    raise RuntimeError("Damping-off beta control failed")
if rows["case0"][0] != 0 or rows["case1"][0] >= 0:
    raise RuntimeError("Radiation switch/loss prerequisite failed")
print(
    f"half-element delta={rows['case1'][0]:.17g}; beta residual={rows['case1'][1] - beta(rows['case1']):.17g}"
)
verdict(
    compare(
        "damping-on beta, relativistic momentum relation",
        rows["case1"][1],
        beta(rows["case1"]),
        atol=3e-16,
    )
)
