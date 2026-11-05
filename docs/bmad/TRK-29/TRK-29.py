"""TRK-29: Near-axis bends suppress opposite-charge deflection

An on-axis positron tracked through an electron-reference sector bend receives no deflection from the native near-axis map. The general map and Runge–Kutta tracking follow the oppositely curved orbit. For a 1 m bend with curvature 0.1 m⁻¹, the native result is horizontal position x = 0 and normalized horizontal momentum = 0 instead of 0.1014 m and 0.1997. Moving the entrance x by only 0.1 μm activates the general map and restores the deflection.

The script tracks on-axis and slightly displaced positrons through a fringe-free bend using native and Runge–Kutta methods. It compares the exit coordinates with a charge-reversed circle intersecting the reference exit plane. Set BMAD_DIST to the Bmad build root.
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
integer j,method
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
v=0
do j=1,2
 if (j==2) v(1)=1e-7_rp
 call init_coord(a,v,lat%ele(1),upstream_end$,particle=positron$)
 do method=1,2
  if (method==1) lat%ele(1)%tracking_method=bmad_standard$
  if (method==2) lat%ele(1)%tracking_method=runge_kutta$
  call track1(a,lat%ele(1),lat%param,b,err_flag=err)
  if (err.or.b%state/=alive$) error stop 'tracking failed'
  write(*,'(a,i0,i0,2es26.17)') 'RESULT case',j,method,b%vec(1:2)
 enddo
enddo
end program
"""
LATTICE = """
no_digested
parameter[geometry]=open
parameter[particle]=electron
beginning[p0c]=1e6
e: sbend,l=1,g=.1,fringe_type=none
lat: line=(e)
use,lat
"""
rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)


def circle(x):
    # Opposite curvature: sin(theta+phi)=(2+g*x)*sin(theta).
    theta, g = 0.1, 0.1
    phi = np.arcsin((2 + g * x) * np.sin(theta)) - theta
    return [
        (2 * np.cos(theta) + g * x * np.cos(theta) - np.cos(phi + theta) - 1) / g,
        np.sin(theta + phi),
    ]


for j, x in enumerate((0.0, 1e-7), 1):
    if compare(f"RK x={x:g}, circle x (m), px/p0", rows[f"case{j}2"], circle(x), atol=2e-10):
        raise RuntimeError("Runge-Kutta circle control failed")
if compare("displaced native, circle x (m), px/p0", rows["case21"], circle(1e-7), atol=2e-14):
    raise RuntimeError("General-map control failed")
verdict(compare("on-axis native, circle x (m), px/p0", rows["case11"], circle(0), atol=2e-14))
