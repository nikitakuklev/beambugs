"""TRK-36: Energy-changing patch Jacobians mix momentum normalizations

A patch that changes reference energy computes geometric matrix terms using both incoming- and outgoing-normalized momenta. Reference renormalization should leave the geometric displacement and its derivative with respect to incoming coordinates unchanged. For a 1 m longitudinal offset and a 100 keV reference-energy increase at 1 MeV/c, x is correctly 0.1857 m, but M₁₆, the derivative of x with respect to relative momentum, is −0.1585 m instead of −0.1761 m. Finite differences also expose the error in a pitched energy-changing patch, while a geometric patch without an energy change passes.

The script compares the pure-offset M₁₆ against the derivative of x = pₓ/pₛ in incoming normalization. It also compares full matrices with central particle differences for geometric and pitched energy-changing patches.
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
real(rp) mat(6,6),vp(6),vm(6),dm(6,6),h
type(coord_struct) ap,am,bp,bm
integer i,j
character(20) filename
do i=1,3
 write(filename,'(a,i0,a)') 'lat',i,'.bmad'
 call bmad_parser(trim(filename),lat,err_flag=err)
 if(err) error stop 'parse failed'
 v=[.002_rp,.003_rp,-.001_rp,.002_rp,.04_rp,.05_rp]
 if(i==1) v=[0.0_rp,.2_rp,0.0_rp,.1_rp,0.0_rp,.1_rp]
 call init_coord(a,v,lat%ele(1),upstream_end$)
 call make_mat6(lat%ele(1),lat%param,a,b,err)
 if (err.or.b%state/=alive$) error stop 'matrix failed'
 mat=lat%ele(1)%mat6
 if (i==1) write(*,'(a,5es26.17)') 'RESULT pure ',a%p0c,b%p0c,b%vec(1),mat(1,6),m_electron
 h=1e-7_rp
 do j=1,6
  vp=v;vm=v;vp(j)=vp(j)+h;vm(j)=vm(j)-h
  call init_coord(ap,vp,lat%ele(1),upstream_end$)
  call init_coord(am,vm,lat%ele(1),upstream_end$)
  call track1(ap,lat%ele(1),lat%param,bp,err_flag=err)
  if (err.or.bp%state/=alive$) error stop 'plus track failed'
  call track1(am,lat%ele(1),lat%param,bm,err_flag=err)
  if (err.or.bm%state/=alive$) error stop 'minus track failed'
  dm(:,j)=(bp%vec-bm%vec)/(2*h)
 enddo
 write(*,'(a,i0,es26.17)') 'RESULT error',i,maxval(abs(dm-mat))
 call deallocate_lat_pointers(lat)
enddo
end program
"""
BASE = "no_digested\nparameter[geometry]=open\nparameter[particle]=electron\nbeginning[p0c]=1e6\n"
ELEMENTS = [
    "e: patch,z_offset=1,e_tot_offset=1e5",
    "e: patch,z_offset=1,x_offset=.01,x_pitch=.01,y_pitch=.02,tilt=.03",
    "e: patch,z_offset=1,x_pitch=.01,e_tot_offset=1e5",
]
files = {
    f"lat{i}.bmad": BASE + element + "\nlat: line=(e)\nuse,lat\n"
    for i, element in enumerate(ELEMENTS, 1)
}
rows = records(fortran(PROGRAM, files=files).log)
ps = np.sqrt(1.1**2 - 0.2**2 - 0.1**2)
x, derivative = 0.2 / ps, -0.2 * 1.1 / ps**3
if compare("pure patch x (m), px/ps", rows["pure"][2], x, atol=2e-15):
    raise RuntimeError("Patch orbit control failed")
if compare("geometric patch maximum Jacobian residual", rows["error2"], [0], atol=1e-9):
    raise RuntimeError("Geometric matrix control failed")
defect = compare("pure patch M16 (m), -px*(1+delta)/ps^3", rows["pure"][3], derivative, atol=2e-14)
defect |= compare("pitched energy patch maximum Jacobian residual", rows["error3"], [0], atol=1e-9)
print(f"reference momenta (eV/c): {rows['pure'][:2].tolist()}")
verdict(defect)
