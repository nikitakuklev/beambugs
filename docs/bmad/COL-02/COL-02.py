"""COL-02: The FFT wrapper ignores mixed transform directions

Bmad’s FFTW wrapper ignores the second and third transform directions when a caller requests a
mixture of forward and inverse axes. An impulse transform with signs (1, 1, −1) has a maximum
error of 2, while the same-sign control agrees to roundoff. In the legacy cathode solver, this
changes a sampled longitudinal field from −512.8 V/m to −1608 V/m, as direct charge-volume
integration establishes. The active cathode route uses a reflected-charge solve and avoids this
transform error.

The script compares an impulse transform with its analytic discrete Fourier transform, then compiles
a legacy cathode-field case. Independent uniform-cell quadrature supplies the electric field
reference; method 2 and the active reflected-charge route are controls.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _collective_fields import cell_field, rows

DFT = r"""
program fft_direction_probe
use fft_interface_mod
use, intrinsic :: iso_fortran_env, only: real64
implicit none
complex(real64) a(4,4,4),b(4,4,4), expected
real(real64) pi,phase,err
integer idir(3),i,j,k,t
pi=acos(-1.0_real64)
a=0;a(2,2,2)=1
do t=1,2
 idir=[1,1,1]
 if(t==2) idir=[1,1,-1]
 call ccfft3d(a,b,idir,4,4,4,0)
 err=0
 do k=1,4
  do j=1,4
   do i=1,4
    phase=2*pi*real(idir(1)*(i-1)+idir(2)*(j-1)+idir(3)*(k-1),real64)/4
    expected=exp(cmplx(0.0_real64,phase,real64))
    err=max(err,abs(b(i,j,k)-expected))
   enddo
  enddo
 enddo
 if(t==1) print '(a,es24.15)', 'RESULT same ',err
 if(t==2) print '(a,es24.15)', 'RESULT mixed ',err
enddo
end program
"""

CATHODE = r"""
program cathode_probe
use open_spacecharge_mod
use, intrinsic :: iso_fortran_env, only: real64
implicit none
real(real64) rho(4,4,4), rr(4,4,4), e(4,4,4,3), b(4,4,4,3), phi(4,4,4), d(3), lo(3)
type(mesh3d_struct) mesh
real(real64), allocatable :: eimage(:,:,:,:)
integer nlo(3),nhi(3),npad(3),j
rho=0; rho(2,2,2)=1e-12_real64
nlo=1;nhi=4;npad=1
d=.001_real64;lo=[-.001_real64,-.001_real64,.001_real64]
mesh%nlo=nlo;mesh%nhi=nhi;mesh%npad=npad
mesh%min=lo;mesh%max=lo+3*d;mesh%delta=d;mesh%gamma=1
allocate(mesh%rho(4,4,4),mesh%efield(4,4,4,3))
mesh%rho=rho
call space_charge_3d(mesh,at_cathode=.true.,image_efield=eimage)
print '(a,3es24.15)','RESULT active ',mesh%efield(3,2,2,:)
do j=1,2
 rr=rho
 call osc_cathodeimages_solver(rr,1.0_real64,lo,d,phi,e,b,nlo,nhi,nlo,nhi,npad,1,1,j)
 if(j==1) print '(a,3es24.15)', 'RESULT method1 ',e(3,2,2,:)
 if(j==2) print '(a,3es24.15)', 'RESULT method2 ',e(3,2,2,:)
enddo
end program
"""


dft = rows(fortran(DFT).log)
if compare("same-axis DFT maximum error", dft["same"], [0], atol=2e-15):
    raise RuntimeError("Same-axis DFT control failed")
defect = compare("mixed-axis DFT maximum error", dft["mixed"], [0], atol=2e-15)
field = rows(fortran(CATHODE).log)


def reference(order):
    return cell_field([0.001, 0, 0.002], [0, 0, 0.002], 1e-12, [0.001] * 3, order) + cell_field(
        [0.001, 0, 0.002], [0, 0, -0.002], -1e-12, [0.001] * 3, order
    )


expected = reference(40)
if not np.allclose(expected, reference(60), rtol=2e-13, atol=2e-10):
    raise RuntimeError("Cell quadrature failed to converge")
# The legacy core uses c²*1e-7; the newer solver uses c²*1.00000000055e-7.
if compare(
    "legacy method 2 field (V/m), image-cell quadrature",
    field["method2"],
    expected / 1.00000000055,
    atol=3e-8,
    rtol=3e-12,
):
    raise RuntimeError("Legacy method 2 control failed")
rounding = float(np.float32(299792458) * np.float32(299792458)) / 299792458.0**2
if compare(
    "active reflected-charge field (V/m), same prefactor precision",
    field["active"],
    expected * rounding,
    atol=3e-8,
    rtol=3e-12,
):
    raise RuntimeError("Active cathode control failed")
defect |= compare(
    "legacy method 1 field (V/m), image-cell quadrature",
    field["method1"],
    expected / 1.00000000055,
    atol=3e-8,
    rtol=3e-12,
)
verdict(defect)
