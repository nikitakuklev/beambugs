---
title: "LAT-15: HDF5 macro-charge weights advertise dimensionless units"
hide:
  - toc
---

# LAT-15 — HDF5 macro-charge weights advertise dimensionless units

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P5 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [hdf5_write_beam weight record, bmad/hdf5/hdf5_write_beam.f90:263–267](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_write_beam.f90#L263-L267)<br>[unit definitions, bmad/hdf5/hdf5_openpmd_mod.f90:19–39](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_openpmd_mod.f90#L19-L39)<br>[hdf5_read_bunch charge readback, bmad/hdf5/hdf5_read_beam.f90:424–427](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L424-L427)<br>[Bmad manual, HDF5 Particle Beam Data Storage (`s:hdf5.beam`), bmad/doc/hdf5-programming.tex:68–82](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/hdf5-programming.tex#L68-L82) |

Writing a beam to HDF5 stores each macroparticle charge in the weight record but labels that record dimensionless. Charges of 1, 2 and 3 pC appear unchanged with unitSI = 1 and seven zero unit exponents. The stored quantity is charge, whose SI dimensions are current multiplied by time, so the metadata contradicts both the supplied charges and the manual. A consumer following the unit metadata cannot interpret these values as charge.

Run `python LAT-15.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) in the same directory.

??? example "Reproducer: LAT-15.py"

    [Download LAT-15.py](LAT-15.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

