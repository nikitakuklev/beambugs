"""LAT-13: HDF5 reading discards the axial-momentum direction sign

Reading a particle beam from HDF5 turns a backward-moving particle into a forward-moving one. The writer preserves the signed axial momentum, but the reader replaces that component with relative momentum deviation and leaves the direction at its default of +1. A three-particle round trip returns +1 instead of −1 for the backward particle while preserving its six phase-space coordinates and identifier. The axial-momentum formula in the manual also omits the direction factor.

The script writes two deterministic electron bunches and reads them back through the public beam-file API. It checks the signed HDF5 momentum and coordinate/identifier controls before comparing the restored direction.
"""

import numpy as np
from _bmad import compare, verdict
from _s3_12_particle_contracts import beam_files, read_beam

rows = read_beam(beam_files()["beam.h5"])
for ib in (1, 2):
    for i in (1, 2, 3):
        row = rows[f"b{ib}p{i}"]
        np.testing.assert_allclose(
            row[:6], np.array([0.001, 0.04, -0.002, -0.02, 0.01, 0.1]) * i, rtol=3e-14, atol=2e-16
        )
        assert row[14] == 100 * ib + i
    assert rows[f"b{ib}p1"][11] == 1
bad = compare("backward-particle direction (signed momentum)", rows["b1p3"][11], -1, atol=0)
print("controls: six coordinates and identifiers preserved; forward direction=+1")
verdict(bad)
