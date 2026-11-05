"""COL-07: Valid Green-function offsets expose removable singularities

The integrated free-space Green functions evaluate singular expressions directly when valid grid
offsets put cell corners on coordinate axes. With cell size h = 1/1024 m and offset (h/2, h/2, 0),
a 4 × 4 × 4 solve returns 128 nonfinite electric-field components and 64 nonfinite potentials. The
cell integrals are finite; direct volume integration gives a sampled field of (3559, 1166, 0) V/m.
Zero and full-cell offsets remain finite.

The script solves the same charged-cell problem at zero, half-cell and full-cell offsets and checks
electric-field and potential finiteness. Independent volume quadrature establishes a finite field
at the half-cell observation point. Set BMAD_DIST to the build root.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _collective_fields import cell_field, rows

PROGRAM = r"""
program green_probe
use open_spacecharge_mod
use, intrinsic :: ieee_arithmetic
use, intrinsic :: iso_fortran_env, only: real64
implicit none
real(real64) :: rho(4,4,4), e(4,4,4,3), phi(4,4,4), delta(3), off(3)
integer j
character(16) label
rho=0; rho(2,2,2)=1e-12_real64
delta=[1.0_real64/1024,1.0_real64/1024,1.0_real64/1024]
do j=0,2
 off=[delta(1)*0.5_real64*j,delta(2)*0.5_real64*j,0.0_real64]
 call osc_freespace_solver2(rho,1.0_real64,delta,efield=e,phi=phi,offset=off)
 write(label,'(a,i0)') 'case',j
 print '(3a,2i6)', 'RESULT ',trim(label),' ',count(.not.ieee_is_finite(e)),count(.not.ieee_is_finite(phi))

enddo
end program
"""


data = rows(fortran(PROGRAM).log)
for j in (0, 2):
    if compare(
        f"offset control {j}: nonfinite E and potential counts", data[f"case{j}"], [0, 0], atol=0
    ):
        raise RuntimeError("Finite-offset control failed")
h = 1 / 1024
expected = cell_field([1.5 * h, 0.5 * h, 0], [0, 0, 0], 1e-12, [h] * 3)
if not np.allclose(
    expected,
    cell_field([1.5 * h, 0.5 * h, 0], [0, 0, 0], 1e-12, [h] * 3, 60),
    rtol=2e-13,
    atol=2e-10,
):
    raise RuntimeError("Finite reference quadrature failed to converge")
print(f"Half-cell offset sample, finite cell-integral E (V/m): {expected.tolist()}")
verdict(
    compare("half-cell offset: nonfinite E and potential counts", data["case1"], [0, 0], atol=0)
)
