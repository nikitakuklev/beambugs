"""COL-38: Beam–beam slices use mid-quantiles instead of documented charge centroids

A beam–beam element places equal-charge Gaussian slices at probability midpoints instead of the charge centroids specified in the manual. For two slices, the positions are ±0.6745 times the bunch RMS length rather than the conditional means ±0.7979 times that length. The offsets are 15.46% smaller than the documented values. A single slice is correctly centered at zero.

The script calls bbi_slice_calc from BMAD_DIST for one and two slices. It compares the two-slice positions with the analytic Gaussian half-bunch mean and checks the centered single-slice control.
"""

import math

import numpy as np
from _bmad import compare, fortran, verdict

PROGRAM = """program probe
use bmad
implicit none
type(ele_struct) ele
real(rp) z(2)
ele%value(sig_z$)=1.0_rp
call bbi_slice_calc(ele,1,z)
write(*,'(2es25.16)') z
call bbi_slice_calc(ele,2,z)
write(*,'(2es25.16)') z
end program
"""
rows = np.array([[float(x) for x in line.split()] for line in fortran(PROGRAM).log.splitlines()])
if rows.shape != (2, 2):
    raise RuntimeError("Expected one- and two-slice results")
if compare("single-slice control and zero padding", rows[0], [0, 0], atol=0):
    raise RuntimeError("Single-slice control failed")
# The conditional mean is integral_0^inf z*phi(z) dz / (1/2) = sqrt(2/pi).
verdict(
    compare(
        "two-slice positions / sigma_z, Gaussian charge centroids",
        rows[1],
        [-math.sqrt(2 / math.pi), math.sqrt(2 / math.pi)],
        atol=1e-5,
    )
)
