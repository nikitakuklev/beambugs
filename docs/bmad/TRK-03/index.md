---
title: "TRK-03: Laboratory-frame field evaluation rotates fixed quadrupole steering"
hide:
  - toc
---

# TRK-03 — Laboratory-frame field evaluation rotates fixed quadrupole steering

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Issue | [[gh:https://github.com/bmad-sim/bmad-ecosystem/issues/2239]] |
| Source | [em_field_calc, steering, bmad/code/em_field_calc.f90:705–723](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L705-L723)<br>[field conversion, bmad/code/em_field_calc.f90:1514–1518](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/em_field_calc.f90#L1514-L1518)<br>[Bmad manual, Hkick, Vkick, and Kick Attributes, bmad/doc/attributes.tex:806–816](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L806-L816) |

Laboratory-frame field evaluation rotates a quadrupole’s steering field with the element tilt, although its horizontal kick must remain horizontal. For a 1 m quadrupole with a 1 GeV/c electron reference, a 1 mrad horizontal kick and π/6 tilt, the returned magnetic field is (−1.668, 2.889, 0) mT instead of (0, 3.336, 0) mT. The steering normalization and an explicit rotation of the body-frame result establish the expected vector. Zero tilt and the quadrupole’s main magnetic field pass their controls.

Run `python TRK-03.py` with [_bmad.py](../helpers/_bmad.py) and [_s3_19_fields_photons_ptc.py](../helpers/_s3_19_fields_photons_ptc.py) in the same directory.

??? example "Reproducer: TRK-03.py"

    [Download TRK-03.py](TRK-03.py)

    ```python
    """TRK-03: Laboratory-frame field evaluation rotates fixed quadrupole steering

    Laboratory-frame field evaluation rotates a quadrupole’s steering field with the element tilt,
    although its horizontal kick must remain horizontal. For a 1 m quadrupole with a 1 GeV/c electron
    reference, a 1 mrad horizontal kick and π/6 tilt, the returned magnetic field is (−1.668, 2.889, 0)
    mT instead of (0, 3.336, 0) mT. The steering normalization and an explicit rotation of the
    body-frame result establish the expected vector. Zero tilt and the quadrupole’s main magnetic field
    pass their controls.

    The script evaluates a zero-gradient quadrupole with hkick=0.001 at zero and both signs of π/6 tilt.
    It compares laboratory and explicitly rotated body fields with the fixed horizontal-steering
    contract, using a magnetic quadrupole control and BMAD_DIST.
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

    d = field_data(5)
    defect = False
    for j in range(3):
        tag = f"F_{j}"
        angle, p0c, length, charge, c_light = d[tag + "_PARAM"]
        body_r = rotate([0.003, 0.002, 0], -angle)
        magnetic_body = p0c / (c_light * charge) * 0.5 * np.array([body_r[1], body_r[0], 0])
        control(d[tag + "_MAG"], rotate(magnetic_body, angle))
        # Quadrupole steering stays horizontal under tilt; q_ref includes its sign.
        expected = np.array([0.0, -0.001 * p0c / (charge * c_light * length), 0.0])
        control(rotate(d[tag + "_BODYB"], angle), expected)
        actual = d[tag + "_LABB"]
        tolerance = 2e-14
        units = "T"
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

