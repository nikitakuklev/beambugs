---
title: "TRK-02: Laboratory-frame electric multipole evaluation cancels element tilt"
hide:
  - toc
---

# TRK-02 — Laboratory-frame electric multipole evaluation cancels element tilt

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [laboratory-to-body coordinates, bmad/code/em_field_calc.f90:223–228](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L223-L228)<br>[electric multipole evaluation, bmad/code/em_field_calc.f90:692–700](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L692-L700)<br>[body-to-laboratory field, bmad/code/em_field_calc.f90:1514–1518](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L1514-L1518)<br>[Bmad manual, Electrostatic Multipole Fields, exiey, bmad/doc/electromagnetic-fields.tex:175–196](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/electromagnetic-fields.tex#L175-L196)<br>[Multipole Attributes: Magnetic and Electric, bmad/doc/attributes.tex:2100–2114](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2100-L2114) |

Laboratory-frame field evaluation of a tilted electric quadrupole cancels the element’s tilt. At (x, y) = (3, 2) mm, an electric gradient of 100 kV/m² tilted by π/6 returns (300.0, −200.0, 0) V/m instead of (323.2, 159.8, 0) V/m. Rotating the coordinates into the element frame and its electric field back to the laboratory establishes the expected vector. Zero tilt and the corresponding magnetic-quadrupole calculation give the correct result.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) · [View source](../helpers/index.md#_s3_19_fields_photons_ptc)

??? example "Reproducer: TRK-02.py"

    [Download TRK-02.py](TRK-02.py)

    ```python
    """TRK-02: Laboratory-frame electric multipole evaluation cancels element tilt

    Laboratory-frame field evaluation of a tilted electric quadrupole cancels the element’s tilt. At (x,
    y) = (3, 2) mm, an electric gradient of 100 kV/m² tilted by π/6 returns (300.0, −200.0, 0) V/m
    instead of (323.2, 159.8, 0) V/m. Rotating the coordinates into the element frame and its electric
    field back to the laboratory establishes the expected vector. Zero tilt and the corresponding
    magnetic-quadrupole calculation give the correct result.

    The script evaluates a pure electric quadrupole at zero and both signs of π/6 tilt in laboratory and
    body frames. It uses explicit Cartesian rotations as the oracle and a magnetic quadrupole at the
    same point as a control; BMAD_DIST selects the build.
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
    from _s3_19_fields_photons_ptc import control, field_data, rotate

    d = field_data(4)
    defect = False
    for j in range(3):
        tag = f"F_{j}"
        angle, p0c, length, charge, c_light = d[tag + "_PARAM"]
        body_r = rotate([0.003, 0.002, 0], -angle)
        magnetic_body = p0c / (c_light * charge) * 0.5 * np.array([body_r[1], body_r[0], 0])
        control(d[tag + "_MAG"], rotate(magnetic_body, angle))
        # E_lab(r)=R(theta) E_body(R(-theta)r); E_body=b1*(x,-y,0).
        body = 1e5 * np.array([body_r[0], -body_r[1], 0])
        control(d[tag + "_BODYE"], body, 2e-11)
        expected = rotate(body, angle)
        actual = d[tag + "_LABE"]
        tolerance = 2e-11
        units = "V/m"
        if j == 0:
            control(actual, expected, tolerance)
        defect |= compare(
            f"tilt={angle:.12g}, lab field ({units}), frame contract", actual, expected, atol=tolerance
        )
    print("Zero tilt, body-field, and magnetic quadrupole controls pass.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

