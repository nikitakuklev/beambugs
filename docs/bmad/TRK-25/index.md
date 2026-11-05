---
title: "TRK-25: Spin Taylor tracking omits spin-frame rotations"
hide:
  - toc
---

# TRK-25 — Spin Taylor tracking omits spin-frame rotations

| | |
|---|---|
| Code | [Bmad](../index.md) · Tracking, RF, fields and spin |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [track1_spin_taylor, bmad/spin/track1_spin_taylor.f90:42–66](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/spin/track1_spin_taylor.f90#L42-L66)<br>[offset_particle optional spin rotation, bmad/code/offset_particle.f90:43–48](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/offset_particle.f90#L43-L48)<br>[Bmad Manual, taylor_map_include_offsets Attribute, `s:mapoff`, bmad/doc/methods.tex:919–934](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/methods.tex#L919-L934) |

A spin Taylor map that excludes element offsets applies its body-frame spin rotation to laboratory-frame spin. For a tilted quadrupole, the orbital coordinates are rotated into and out of the body frame but the spin is not. A 0.6 m quadrupole tilted by 0.43 rad produces horizontal spin 0.2991 instead of 0.2998 in the example. Explicit frame rotations and a freshly generated map including offsets agree, and the zero-tilt control passes.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _rf_and_spin.py](../helpers/_rf_and_spin.py) · [View source](../helpers/index.md#_rf_and_spin)

??? example "Reproducer: TRK-25.py"

    [Download TRK-25.py](TRK-25.py)

    ```python
    """TRK-25: Spin Taylor tracking omits spin-frame rotations

    A spin Taylor map that excludes element offsets applies its body-frame spin rotation to laboratory-frame spin. For a tilted quadrupole, the orbital coordinates are rotated into and out of the body frame but the spin is not. A 0.6 m quadrupole tilted by 0.43 rad produces horizontal spin 0.2991 instead of 0.2998 in the example. Explicit frame rotations and a freshly generated map including offsets agree, and the zero-tilt control passes.

    The script generates fresh second-order PTC spin maps with offsets included and excluded, using BMAD_DIST. It evaluates the stored quaternion polynomial independently and applies explicit input/output spin-frame rotations; an untilted quadrupole is the control.
    """

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _rf_and_spin import (
        TAYLOR_LATTICE,
        TAYLOR_PROGRAM,
        control,
        evaluate,
        rotate,
        spin,
        spin_data,
    )

    d, maps = spin_data(fortran(TAYLOR_PROGRAM, ["2"], files={"verify.bmad": TAYLOR_LATTICE}).log)
    defect = False
    for i in range(2):
        local, lab = f"P_{i}_0", f"P_{i}_1"
        angle = float(d[local + "_ANGLE"][0])
        for tag in (local, lab):
            control(
                f"{tag} orbital and spin expansion origins",
                np.r_[d[tag + "_ORBITREF"], d[tag + "_SPINREF"]],
                np.zeros(12),
                atol=0,
            )
        incoming = d[local + "_IN"].copy()
        for indices in ([0, 2], [1, 3]):
            incoming[indices] = rotate([*incoming[indices], 0], -angle)[:2]
        q = evaluate(maps[local], incoming, np.zeros(6))
        initial = d[local + "_S0"]
        # A body-frame spin map transforms as R(theta) M_body R(-theta).
        expected = rotate(spin(q, rotate(initial, -angle)), angle)
        control(
            f"tilt={angle} included-offset spin, explicit frame conjugation",
            d[lab + "_ACTUAL"],
            expected,
            atol=2e-14,
        )
        control(f"tilt={angle} orbital agreement", d[local + "_ORBIT"], d[lab + "_ORBIT"], atol=2e-13)
        if i == 0:
            control("zero-tilt excluded-offset spin", d[local + "_ACTUAL"], expected, atol=2e-14)
        else:
            defect |= compare(
                "tilted excluded-offset spin, explicit frame conjugation",
                d[local + "_ACTUAL"],
                expected,
                atol=2e-14,
            )
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

