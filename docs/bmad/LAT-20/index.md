---
title: "LAT-20: The HDF5 manual places species metadata one group too high"
hide:
  - toc
---

# LAT-20 — The HDF5 manual places species metadata one group too high

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P4 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [Bmad manual, HDF5 Particle Beam Data Storage (`s:hdf5.beam`), bmad/doc/hdf5-programming.tex:29–42](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/hdf5-programming.tex#L29-L42)<br>[hdf5_write_beam species subgroup, bmad/hdf5/hdf5_write_beam.f90:157–167](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_write_beam.f90#L157-L167) |

Writing an HDF5 beam stores species and charge metadata in a species subgroup below each particles group. The manual instead places speciesType, numParticles, totalCharge and chargeLive directly in the particles group. A three-electron record stores these attributes under /data/00001/particles/electron/, with total charge 6 pC and live charge 4 pC. Readers following the documented path look one group too high.

Run `python LAT-20.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) in the same directory.

??? example "Reproducer: LAT-20.py"

    [Download LAT-20.py](LAT-20.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

