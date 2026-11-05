"""TRK-02: Laboratory-frame electric multipole evaluation cancels element tilt

Laboratory-frame field evaluation of a tilted electric quadrupole cancels the element’s tilt. At (x,
y) = (3, 2) mm, an electric gradient of 100 kV/m² tilted by π/6 returns (300.0, −200.0, 0) V/m
instead of (323.2, 159.8, 0) V/m. Rotating the coordinates into the element frame and its electric
field back to the laboratory establishes the expected vector. Zero tilt and the corresponding
magnetic-quadrupole calculation give the correct result.

The script evaluates a pure electric quadrupole at zero and both signs of π/6 tilt in laboratory and
body frames. It uses explicit Cartesian rotations as the oracle and a magnetic quadrupole at the
same point as a control; BMAD_DIST selects the build.
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

d = field_data(4)
defect = False
for j in range(3):
    tag = f"F_{j}"
    angle, p0c, length, charge, c_light = d[tag + "_PARAM"]
    body_r = rotate([0.003, 0.002, 0], -angle)
    magnetic_body = p0c / (c_light * charge) * 0.5 * np.array([body_r[1], body_r[0], 0])
    control(d[tag + "_MAG"], rotate(magnetic_body, angle))
    # E_lab(r)=R(theta) E_body(R(-theta)r); E_body=b1*(x,-y,0).
    body = 1e5 * np.array([body_r[0], -body_r[1], 0])
    control(d[tag + "_BODYE"], body, 2e-11)
    expected = rotate(body, angle)
    actual = d[tag + "_LABE"]
    tolerance = 2e-11
    units = "V/m"
    if j == 0:
        control(actual, expected, tolerance)
    defect |= compare(
        f"tilt={angle:.12g}, lab field ({units}), frame contract", actual, expected, atol=tolerance
    )
print("Zero tilt, body-field, and magnetic quadrupole controls pass.")
verdict(defect)
