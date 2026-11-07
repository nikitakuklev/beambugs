---
title: "LAT-14: Ordinary HDF5 reads leave the live-particle count unset"
hide:
  - toc
---

# LAT-14 — Ordinary HDF5 reads leave the live-particle count unset

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [hdf5_read_bunch initialization, bmad/hdf5/hdf5_read_beam.f90:321–327](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L321-L327)<br>[ordinary read completion, bmad/hdf5/hdf5_read_beam.f90:562–587](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L562-L587)<br>[patch splitting, bmad/hdf5/hdf5_read_beam.f90:219–231](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L219-L231) |

Reading ordinary HDF5 species records restores particle states and charge totals but leaves the bunch live-particle count at zero. A bunch with states [alive, lost, alive] therefore reports zero live particles instead of two, even though its live charge is restored correctly. Reading the same particles from ASCII or HDF5 particle patches gives the correct count. The count must agree with the number of particles whose state is alive.

Run `python LAT-14.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) in the same directory.

??? example "Reproducer: LAT-14.py"

    [Download LAT-14.py](LAT-14.py)

    ```python
    """LAT-14: Ordinary HDF5 reads leave the live-particle count unset

    Reading ordinary HDF5 species records restores particle states and charge totals but leaves the bunch live-particle count at zero. A bunch with states [alive, lost, alive] therefore reports zero live particles instead of two, even though its live charge is restored correctly. Reading the same particles from ASCII or HDF5 particle patches gives the correct count. The count must agree with the number of particles whose state is alive.

    The script writes two three-particle bunches in ordinary HDF5, patch HDF5 and ASCII formats. It checks states and charge totals, then compares each live count with the count derived directly from particle states.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _s3_12_particle_contracts import beam_files, read_beam

    files = beam_files()
    bad = False
    for name in ("beam.txt", "patches.h5", "beam.h5"):
        rows = read_beam(files[name], name)
        for ib in (1, 2):
            states = [rows[f"b{ib}p{i}"][12] for i in (1, 2, 3)]
            np.testing.assert_array_equal(states, [1, 2, 1])
            np.testing.assert_allclose(rows[f"b{ib}"][:2], [6e-12, 4e-12], rtol=2e-15)
            mismatch = compare(
                f"{name} bunch {ib} n_live (count state=alive)",
                rows[f"b{ib}"][2],
                states.count(1),
                atol=0,
            )
            if name != "beam.h5" and mismatch:
                raise RuntimeError("Live-count control failed")
            bad |= mismatch
    verdict(bad)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

