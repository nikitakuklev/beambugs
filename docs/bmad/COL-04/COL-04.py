"""COL-04: S-based FFT kicks omit the test particle's charge state

For homogeneous bunches of multiply charged ions, s-based FFT space charge omits the test
particle’s charge state from the field-to-momentum conversion. At fixed deposited charge, mass and
kinematics, a charge +2 particle must receive twice the +1 impulse by the Lorentz force. Equal-mass
species instead both receive a normalized horizontal kick of 3.901e−9; the +2 expectation is
7.802e−9. The +1 control agrees with the force calculation.

The script deposits identical charge distributions for equal-mass +1 and +2 species and gathers the
electric field before comparing the normalized impulse with the Lorentz-force factor
|Z|EₓΔs/(p₀cβγ²). The +1 case controls field gathering and units.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _collective_fields import rows

PROGRAM = r"""
program fft_species_probe
use csr_and_space_charge_mod
implicit none
type(csr_struct) c
type(coord_struct) p(8), p0(8)
real(rp) ex(3),factor,expected,actual
integer i,j
character(16) label
c%mesh3d%nhi=[8,8,8]; c%mesh3d%gamma=2
c%actual_track_step=.01_rp
do j=1,2
 do i=1,8
  p(i)=coord_struct()
  p(i)%species=species_id('@M3.0+')
  if(j==2) p(i)%species=species_id('@M3.0+2')
  p(i)%state=alive$
  p(i)%vec(1)=merge(-.001_rp,.001_rp,btest(i-1,0))
  p(i)%vec(3)=merge(-.001_rp,.001_rp,btest(i-1,1))
  p(i)%vec(5)=merge(-.001_rp,.001_rp,btest(i-1,2))
  p(i)%charge=1e-12_rp
  p(i)%p0c=mass_of(p(i)%species)*sqrt(3.0_rp)
  p(i)%beta=sqrt(.75_rp)
 enddo
 p0=p
 call apply_fft_3d_kicks(c,p)
 call interpolate_field(p0(1)%vec(1),p0(1)%vec(3),p0(1)%vec(5),c%mesh3d,E=ex)
 factor=c%actual_track_step/(p0(1)%p0c*p0(1)%beta)
 expected=abs(charge_of(p(1)%species))*ex(1)*factor/c%mesh3d%gamma**2
 actual=p(1)%vec(2)
 write(label,'(a,i0)') 'charge',j
 print '(3a,3es24.15)', 'RESULT ',trim(label),' ',actual,expected,actual/expected
enddo
end program
"""


data = rows(fortran(PROGRAM).log)
single, double = data["charge1"], data["charge2"]
if compare(
    "+1 normalized horizontal kick, Lorentz force", single[0], single[1], atol=1e-23, rtol=2e-13
):
    raise RuntimeError("Unit-charge force control failed")
if not np.allclose(double[1], 2 * single[1], rtol=2e-13, atol=0):
    raise RuntimeError("Equal-field, mass and kinematics control failed")
verdict(
    compare(
        "+2 normalized horizontal kick, twice the +1 force",
        double[0],
        double[1],
        atol=1e-23,
        rtol=2e-13,
    )
)
