"""TRK-01: Grid-map superposition duplicates transverse fields on axis

When a Cartesian grid precedes an axisymmetric grid in the same element, evaluating the field
exactly on axis adds the first grid’s transverse field twice. Constant magnetic grids (0.7, −0.4,
0.2) T and (0, 0, 1.3) T return (1.4, −0.8, 1.5) T instead of their vector sum (0.7, −0.4, 1.5) T.
Reversing the grid order, evaluating either grid alone, or moving off axis gives the correct result.
The axisymmetric branch leaves transverse components from the preceding grid in its temporary
accumulator.

The script constructs constant Cartesian and rotationally symmetric magnetic grids, evaluates them
separately and in both orders, and checks radii 0, 1e-9 and 1e-3 m. It compares the combined field
with direct vector addition using BMAD_DIST.
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
from _s3_19_fields_photons_ptc import control, field_data

d = field_data(1)
a, b = np.array([0.7, -0.4, 0.2]), np.array([0, 0, 1.3])
for j in range(3):
    control(d[f"G_1_{j}"], a)
    control(d[f"G_2_{j}"], b)
    control(d[f"G_4_{j}"], a + b)
    if j:
        control(d[f"G_3_{j}"], a + b)
print("Single maps, reversed order, and r=1e-9,1e-3 m controls pass at 2e-14 T.")
verdict(compare("On-axis A+B (T), vector sum", d["G_3_0"], a + b, atol=2e-14))
