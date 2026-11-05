"""LAT-20: The HDF5 manual places species metadata one group too high

Writing an HDF5 beam stores species and charge metadata in a species subgroup below each particles group. The manual instead places speciesType, numParticles, totalCharge and chargeLive directly in the particles group. A three-electron record stores these attributes under /data/00001/particles/electron/, with total charge 6 pC and live charge 4 pC. Readers following the documented path look one group too high.

The script writes a deterministic beam and checks both the particles group and its electron subgroup with h5py. BMAD_SOURCE supplies the documented group path and attribute list.
"""

import io

import h5py
import numpy as np
from _bmad import verdict
from _s3_12_particle_contracts import beam_files, document_claim

with h5py.File(io.BytesIO(beam_files()["beam.h5"]), "r") as file:
    parent = file["data/00001/particles"]
    record = parent["electron"]
    names = ("speciesType", "numParticles", "totalCharge", "chargeLive")
    assert all(name in record.attrs and name not in parent.attrs for name in names)
    assert record.attrs["numParticles"] == 3
    np.testing.assert_allclose(
        [
            np.asarray(record.attrs["totalCharge"]).item(),
            np.asarray(record.attrs["chargeLive"]).item(),
        ],
        [6e-12, 4e-12],
        rtol=2e-15,
    )
    print(
        f"metadata group={record.name}; parent has none of {names}; particles=3, totalCharge=6e-12 C, chargeLive=4e-12 C"
    )
verdict(
    document_claim(
        "bmad/doc/hdf5-programming.tex",
        r"/data/%T/particles/ \end{example} where \vn{%T} is an integer.",
    )
)
