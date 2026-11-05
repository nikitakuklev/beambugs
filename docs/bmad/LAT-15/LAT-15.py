"""LAT-15: HDF5 macro-charge weights advertise dimensionless units

Writing a beam to HDF5 stores each macroparticle charge in the weight record but labels that record dimensionless. Charges of 1, 2 and 3 pC appear unchanged with unitSI = 1 and seven zero unit exponents. The stored quantity is charge, whose SI dimensions are current multiplied by time, so the metadata contradicts both the supplied charges and the manual. A consumer following the unit metadata cannot interpret these values as charge.

The script writes deterministic charges and opens the generated HDF5 file with h5py. It verifies the input charges and total-charge metadata before comparing the weight unit exponents with Coulomb dimensions.
"""

import io

import h5py
import numpy as np
from _bmad import compare, verdict
from _s3_12_particle_contracts import beam_files, values

with h5py.File(io.BytesIO(beam_files()["beam.h5"]), "r") as file:
    record = file["data/00001/particles/electron"]
    charge = values(record, "weight")
    np.testing.assert_allclose(charge, [1e-12, 2e-12, 3e-12], rtol=2e-15)
    np.testing.assert_allclose(record.attrs["totalCharge"], charge.sum(), rtol=2e-15)
    assert record.attrs["chargeUnitSI"] == record["weight"].attrs["unitSI"] == 1
    print(
        f"weight (C)={charge.tolist()}; totalCharge={np.asarray(record.attrs['totalCharge']).item():.16g}; unitSI=1"
    )
    # SI base dimensions are length, mass, time, current, temperature, amount, luminosity.
    # One Coulomb = one ampere second.
    bad = compare(
        "weight unitDimension (C=A*s)",
        record["weight"].attrs["unitDimension"],
        [0, 0, 1, 1, 0, 0, 0],
        atol=0,
    )
verdict(bad)
