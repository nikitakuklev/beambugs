"""COL-12: CSR3D boundary derivatives use integer division

The experimental three-dimensional coherent synchrotron radiation solver uses integer division in
its one-sided density-derivative stencils. The coefficients 3/2 and 1/2 become 1 and 0, so even a
constant density acquires nonzero endpoint derivatives. For f(z) = 1 + 0.3z + 0.7z² on a grid of
spacing 0.2, the endpoint derivatives are 5.880 and −5.880 instead of 0.3000 and 1.420. Interior
derivatives and zero FFT padding are correct.

The script differentiates a quadratic density on five evenly spaced points and compares against its
analytic derivative. It requires the interior stencil and doubled-grid padding to pass before
checking both endpoints. Set BMAD_DIST to the build root.
"""

import numpy as np
from _bmad import compare, fortran, verdict
from _collective_fields import rows

PROGRAM = r"""

program csr3d_derivative
  use bmad
  use csr3d_mod, only: calc_density_derivative_complex
  implicit none
  real(rp) density(1,1,5), z
  complex(rp) derivative(2,2,10)
  integer i
  do i = 1, 5
    z = (i-1)*0.2_rp
    density(1,1,i) = 1 + 0.3_rp*z + 0.7_rp*z*z
  enddo
  call calc_density_derivative_complex(density, derivative, 0.2_rp)
  write(*,'(a,5es25.16e3)') 'RESULT derivative ', real(derivative(1,1,1:5),rp)
  derivative(1,1,1:5) = 0
  write(*,'(a,es25.16e3)') 'RESULT padding ', maxval(abs(derivative))
end program
"""


data = rows(fortran(PROGRAM).log)
actual = data["derivative"]
expected = 0.3 + 1.4 * np.arange(5) * 0.2
if not np.array_equal(data["padding"], [0]):
    raise RuntimeError("Zero-padding control failed")
if compare(
    "interior density derivatives, analytic quadratic derivative",
    actual[1:-1],
    expected[1:-1],
    atol=2e-15,
    rtol=2e-15,
):
    raise RuntimeError("Interior-stencil control failed")
defect = compare(
    "endpoint density derivatives, analytic quadratic derivative",
    actual[[0, -1]],
    expected[[0, -1]],
    atol=3e-15,
    rtol=3e-15,
)
z = np.arange(5) * 0.2
density = 1 + 0.3 * z + 0.7 * z * z
# Integer division removes the third sample and changes each 3/2 to 1.
integer_ends = np.array([-density[0] + 2 * density[1], -2 * density[-2] + density[-1]]) / 0.2
if defect and not np.allclose(actual[[0, -1]], integer_ends, rtol=3e-15, atol=3e-15):
    raise RuntimeError("Endpoint mismatch does not match the integer-division stencil")
verdict(defect)
