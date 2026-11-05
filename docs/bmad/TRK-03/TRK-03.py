"""TRK-03: Laboratory-frame field evaluation rotates fixed quadrupole steering

Laboratory-frame field evaluation rotates a quadrupole’s steering field with the element tilt,
although its horizontal kick must remain horizontal. For a 1 m quadrupole with a 1 GeV/c electron
reference, a 1 mrad horizontal kick and π/6 tilt, the returned magnetic field is (−1.668, 2.889, 0)
mT instead of (0, 3.336, 0) mT. The steering normalization and an explicit rotation of the
body-frame result establish the expected vector. Zero tilt and the quadrupole’s main magnetic field
pass their controls.

The script evaluates a zero-gradient quadrupole with hkick=0.001 at zero and both signs of π/6 tilt.
It compares laboratory and explicitly rotated body fields with the fixed horizontal-steering
contract, using a magnetic quadrupole control and BMAD_DIST.
"""

import os

# Set numerical pool sizes before NumPy/SciPy are imported.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"


import numpy as np
from _bmad import compare, verdict
from _s3_19_fields_photons_ptc import control, field_data, rotate

d = field_data(5)
defect = False
for j in range(3):
    tag = f"F_{j}"
    angle, p0c, length, charge, c_light = d[tag + "_PARAM"]
    body_r = rotate([0.003, 0.002, 0], -angle)
    magnetic_body = p0c / (c_light * charge) * 0.5 * np.array([body_r[1], body_r[0], 0])
    control(d[tag + "_MAG"], rotate(magnetic_body, angle))
    # Quadrupole steering stays horizontal under tilt; q_ref includes its sign.
    expected = np.array([0.0, -0.001 * p0c / (charge * c_light * length), 0.0])
    control(rotate(d[tag + "_BODYB"], angle), expected)
    actual = d[tag + "_LABB"]
    tolerance = 2e-14
    units = "T"
    if j == 0:
        control(actual, expected, tolerance)
    defect |= compare(
        f"tilt={angle:.12g}, lab field ({units}), frame contract", actual, expected, atol=tolerance
    )
print("Zero tilt, body-field, and magnetic quadrupole controls pass.")
verdict(defect)
