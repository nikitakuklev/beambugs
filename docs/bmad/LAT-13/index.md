---
title: "LAT-13: HDF5 reading discards the axial-momentum direction sign"
hide:
  - toc
---

# LAT-13 — HDF5 reading discards the axial-momentum direction sign

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [hdf5_read_bunch reads signed momentum, bmad/hdf5/hdf5_read_beam.f90:389–395](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L389-L395)<br>[momentum conversion, bmad/hdf5/hdf5_read_beam.f90:562–585](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L562-L585)<br>[hdf5_write_beam, bmad/hdf5/hdf5_write_beam.f90:203–209](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_write_beam.f90#L203-L209)<br>[coord_struct direction contract, bmad/modules/bmad_struct.f90:584–586](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/bmad_struct.f90#L584-L586)<br>[Bmad manual, HDF5 Particle Beam Data Storage (`s:hdf5.beam`), bmad/doc/hdf5-programming.tex:121–129](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/hdf5-programming.tex#L121-L129) |

Reading a particle beam from HDF5 turns a backward-moving particle into a forward-moving one. The writer preserves the signed axial momentum, but the reader replaces that component with relative momentum deviation and leaves the direction at its default of +1. A three-particle round trip returns +1 instead of −1 for the backward particle while preserving its six phase-space coordinates and identifier. The axial-momentum formula in the manual also omits the direction factor.

Run `python LAT-13.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) in the same directory.

??? example "Reproducer: LAT-13.py"

    [Download LAT-13.py](LAT-13.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

