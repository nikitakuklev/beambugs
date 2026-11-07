"""TRK-30: Solenoid strength counts tracking charge twice

A solenoid with an electron reference rotates equal-mass electrons and positrons with the same handedness in native tracking. The relative charge factor already contains the tracked charge sign, and multiplying by the tracked charge again cancels that reversal. For a 1 m solenoid of strength 1 m⁻¹ and entrance x = 1 mm, both species leave at y = −0.4207 mm; the positron should leave at +0.4207 mm. The analytic solenoid map and Runge–Kutta tracking agree on the required sign.

The script tracks electrons and positrons through a fringe-free solenoid using both native and Runge–Kutta methods. It derives the transverse positions from the hard-edge solenoid body solution at fixed canonical entrance momentum.
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
integer species,method
call bmad_parser('lat.bmad',lat,err_flag=err)
if (err) error stop 'parse failed'
bmad_com%rel_tol_adaptive_tracking=1e-13_rp
bmad_com%abs_tol_adaptive_tracking=1e-16_rp
! Preserve the historical default-real literal, then convert it to rp.
v=0; v(1)=0.001
do species=1,2
 if (species==1) call init_coord(a,v,lat%ele(1),upstream_end$,particle=electron$)
 if (species==2) call init_coord(a,v,lat%ele(1),upstream_end$,particle=positron$)
 do method=1,2
  if (method==1) lat%ele(1)%tracking_method=bmad_standard$
  if (method==2) lat%ele(1)%tracking_method=runge_kutta$
  call track1(a,lat%ele(1),lat%param,b,err_flag=err)
  if (err.or.b%state/=alive$) error stop 'tracking failed'
  write(*,'(a,i0,i0,2es26.17)') 'RESULT case',species,method,b%vec(1),b%vec(3)
 enddo
enddo
end program
"""
LATTICE = """
no_digested
parameter[geometry]=open
parameter[particle]=electron
beginning[p0c]=1e6
e: solenoid,l=1,ks=1,fringe_type=none
lat: line=(e)
use,lat
"""
rows = records(fortran(PROGRAM, files={"lat.bmad": LATTICE}).log)
x0 = float(np.float32(0.001))  # The original Fortran input was a default-real literal.
theta = 0.5 / np.sqrt(1 - (x0 / 2) ** 2)
# ks changes sign with charge, while pz and cos(theta)^2 do not.
electron = np.array([np.cos(theta) ** 2 * x0, -np.sin(theta) * np.cos(theta) * x0])
positron = electron * [1, -1]
for key, expected in (("case11", electron), ("case12", electron), ("case22", positron)):
    if compare(f"{key} control x,y (m), analytic solenoid", rows[key], expected, atol=2e-15):
        raise RuntimeError("Solenoid control failed")
verdict(compare("native positron x,y (m), analytic solenoid", rows["case21"], positron, atol=2e-15))
