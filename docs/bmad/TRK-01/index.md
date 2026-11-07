---
title: "TRK-01: Grid-map superposition duplicates transverse fields on axis"
hide:
  - toc
---

# TRK-01 — Grid-map superposition duplicates transverse fields on axis

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [em_field_calc, on-axis branch, bmad/code/em_field_calc.f90:1381–1399](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L1381-L1399)<br>[field accumulation, bmad/code/em_field_calc.f90:1427–1430](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L1427-L1430)<br>[Bmad manual, Field Maps, bmad/doc/attributes.tex:2169–2178](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2169-L2178) |

When a Cartesian grid precedes an axisymmetric grid in the same element, evaluating the field exactly on axis adds the first grid’s transverse field twice. Constant magnetic grids (0.7, −0.4, 0.2) T and (0, 0, 1.3) T return (1.4, −0.8, 1.5) T instead of their vector sum (0.7, −0.4, 1.5) T. Reversing the grid order, evaluating either grid alone, or moving off axis gives the correct result. The axisymmetric branch leaves transverse components from the preceding grid in its temporary accumulator.

Run `python TRK-01.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-01.py"

    [Download TRK-01.py](TRK-01.py)

    ```python
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

