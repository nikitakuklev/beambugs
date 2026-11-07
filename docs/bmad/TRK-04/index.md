---
title: "TRK-04: Planar-wiggler longitudinal field derivatives disagree with the field"
hide:
  - toc
---

# TRK-04 — Planar-wiggler longitudinal field derivatives disagree with the field

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [em_field_calc, planar wiggler, bmad/code/em_field_calc.f90:735–761](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L735-L761)<br>[Bmad manual, Periodic Type Wigglers, bbkykz, bmad/doc/elements.tex:5664–5675](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L5664-L5675) |

Planar-wiggler field evaluation returns incorrect derivatives of the longitudinal magnetic field. For a 1 T wiggler with 0.2 m period and horizontal wave number 13 m⁻¹, at transverse position (x, y) = (3, 2) mm and longitudinal position s = 370 mm the returned derivatives are (0.4110, −1.870, 18.49) T/m instead of (0.02579, −25.46, 1.160) T/m. Analytic differentiation and fourth-order finite differences of the returned field agree on the expected values. The field itself and the derivatives of its two transverse components are correct.

Run `python TRK-04.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-04.py"

    [Download TRK-04.py](TRK-04.py)

    ```python
    """TRK-04: Planar-wiggler longitudinal field derivatives disagree with the field

    Planar-wiggler field evaluation returns incorrect derivatives of the longitudinal magnetic field.
    For a 1 T wiggler with 0.2 m period and horizontal wave number 13 m⁻¹, at transverse position (x, y)
    = (3, 2) mm and longitudinal position s = 370 mm the returned derivatives are (0.4110, −1.870,
    18.49) T/m instead of (0.02579, −25.46, 1.160) T/m. Analytic differentiation and fourth-order finite
    differences of the returned field agree on the expected values. The field itself and the derivatives
    of its two transverse components are correct.

    The script samples the resolved planar-wiggler field at horizontal wave numbers 0 and 13 m⁻¹ and
    requests its analytic derivative matrix. It compares that matrix with explicit differentiation and a
    fourth-order central stencil with 1e-5 m steps;
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


    import math

    import numpy as np
    from _bmad import compare, verdict
    from _s3_19_fields_photons_ptc import control, field_data

    d = field_data(6)
    defect = False
    # Differentiate the resolved B field; a fourth-order central stencil is a second oracle.
    for i in range(2):
        tag = f"W_{i}"
        kx, period, amp, length, x, y, s, h = d[tag + "_PARAM"]
        kz = 2 * math.pi / period
        ky = math.hypot(kx, kz)
        cx, sx = math.cos(kx * x), math.sin(kx * x)
        ch, sh = math.cosh(ky * y), math.sinh(ky * y)
        cz, sz = math.cos(kz * (s - length / 2)), math.sin(kz * (s - length / 2))
        field = amp * np.array([-kx / ky * sx * sh * cz, cx * ch * cz, -kz / ky * cx * sh * sz])
        jac = amp * np.array(
            [
                [
                    -kx * kx / ky * cx * sh * cz,
                    -kx * sx * ch * cz,
                    kx * kz / ky * sx * sh * sz,
                ],
                [-kx * sx * ch * cz, ky * cx * sh * cz, -kz * cx * ch * sz],
                [
                    kx * kz / ky * sx * sh * sz,
                    -kz * cx * ch * sz,
                    -kz * kz / ky * cx * sh * cz,
                ],
            ]
        )
        control(d[tag + "_B"], field)
        actual = np.column_stack([d[f"{tag}_D_{j}"] for j in (1, 2, 3)])
        fd = np.column_stack(
            [
                (
                    d[f"{tag}_S_{j}_1"]
                    - 8 * d[f"{tag}_S_{j}_2"]
                    + 8 * d[f"{tag}_S_{j}_3"]
                    - d[f"{tag}_S_{j}_4"]
                )
                / (12 * h)
                for j in (1, 2, 3)
            ]
        )
        fd_error = control(fd, jac, 5e-10)
        control(actual[:2], jac[:2])
        assert abs(np.trace(jac)) < 1e-14
        defect |= compare(f"kx={kx:g}, dBz (T/m), analytic derivative", actual[2], jac[2], atol=5e-10)
        print(
            f"Finite-difference check kx={kx:g}: returned dBz={actual[2]}; expected={jac[2]}; FD residual={fd_error:.3e} T/m"
        )


    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

