---
title: "LAT-11: An omitted HDF5 momentum policy selects inconsistent defaults"
hide:
  - toc
---

# LAT-11 — An omitted HDF5 momentum policy selects inconsistent defaults

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`hdf5_read_bunch`, bmad/hdf5/hdf5_read_beam.f90:564–584](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L564-L584)<br>[public momentum-policy contract, bmad/hdf5/hdf5_read_beam.f90:7–24](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L7-L24)<br>[The momentum-shift diagnostics, bmad/hdf5/hdf5_read_beam.f90:538–551](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/hdf5/hdf5_read_beam.f90#L538-L551) |

Reading an HDF5 beam at a different reference momentum without specifying conserve_momentum combines incompatible defaults. Changing the reference from 1 to 1.7 MeV/c keeps the relative total-momentum deviation δ = 0.1 but changes transverse momenta divided by the reference momentum from (0.04, −0.02) to (0.02353, −0.01176). The documented default is false, which retains all three normalized momentum coordinates. Explicit false preserves those coordinates; the source’s warning branches describe the opposite momentum policy.

Run `python LAT-11.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_11_particle_io.py](../helpers/_s3_11_particle_io.py) in the same directory.

??? example "Reproducer: LAT-11.py"

    [Download LAT-11.py](LAT-11.py)

    ```python
    """LAT-11: An omitted HDF5 momentum policy selects inconsistent defaults

    Reading an HDF5 beam at a different reference momentum without specifying conserve_momentum combines incompatible defaults. Changing the reference from 1 to 1.7 MeV/c keeps the relative total-momentum deviation δ = 0.1 but changes transverse momenta divided by the reference momentum from (0.04, −0.02) to (0.02353, −0.01176). The documented default is false, which retains all three normalized momentum coordinates. Explicit false preserves those coordinates; the source’s warning branches describe the opposite momentum policy.

    The script writes one electron at a 1 MeV/c reference and reads it at 1.7 MeV/c with omitted, false and true policies in fresh processes. It compares omission with explicit false for the three normalized momentum components and validates explicit true against conservation of physical momentum.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _s3_11_particle_io import policy

    results = {mode: policy(mode) for mode in ("true", "false", "omitted")}
    for mode, result in results.items():
        if compare(
            f"{mode} input control: x, px, y, py, z, delta, reference",
            result["input"][:7],
            [0.001, 0.04, -0.002, -0.02, 0.01, 0.1, 1e6],
            atol=2e-15,
        ):
            raise RuntimeError("Input control failed")
        if compare(
            f"{mode} position/reference/time control",
            result["output"][[0, 2, 6, 8]],
            [0.001, -0.002, 1.7e6, 4.1e-9],
            atol=2e-18,
            rtol=3e-15,
        ):
            raise RuntimeError("Read control failed")
    expected = np.array([0.04, -0.02, 0.1])
    if compare(
        "explicit false normalized momenta; documented policy",
        results["false"]["output"][[1, 3, 5]],
        expected,
        atol=2e-15,
    ):
        raise RuntimeError("Explicit-false control failed")
    true_expected = [0.04 / 1.7, -0.02 / 1.7, 1.1 / 1.7 - 1]
    if compare(
        "explicit true normalized momenta; physical momentum conserved",
        results["true"]["output"][[1, 3, 5]],
        true_expected,
        atol=2e-15,
    ):
        raise RuntimeError("Explicit-true control failed")
    verdict(
        compare(
            "omitted normalized momenta; documented false default",
            results["omitted"]["output"][[1, 3, 5]],
            expected,
            atol=2e-15,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

