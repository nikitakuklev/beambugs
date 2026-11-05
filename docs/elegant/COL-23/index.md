---
title: "COL-23: TRWAKE quadrupole probe coordinates omit element misalignment"
hide:
  - toc
---

# COL-23 — TRWAKE quadrupole probe coordinates omit element misalignment

| | |
|---|---|
| Code | [Elegant](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant master (`0355dd6`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [tracking caller, do_tracking.c:2483–2486](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/do_tracking.c#L2483-L2486)<br>[trwake.c:154–156](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L154-L156)<br>[binTransverseTimeDistribution, trwake.c:649–660](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L649-L660)<br>[gather call, trwake.c:251–255](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L251-L255)<br>[`applyTransverseWakeKicks`, trwake.c:328–331](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L328-L331)<br>[trwake.c:151–152](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/trwake.c#L151-L152)<br>[quadrupole wake, trwake.tex:53–58](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/trwake.tex#L53-L58)<br>[misalignment parameters, track_data.c:2604–2605](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L2604-L2605) |

TRWAKE quadrupole wakes (probe exponents 1, drive exponents 0) use each particle's global transverse coordinate even when the structure has misalignments DX and DY. The kick must use displacement from the structure, so translating beam and structure together must preserve the kicks, and a beam on the structure's axis must receive none. For a 0.6000 nC electron bunch represented by 18 macroparticles, a common shift of (3.100, −1.700) mm raises the trailing particle's horizontal slope kick from 1.498×10⁻⁵ to 6.660×10⁻⁵, and an on-axis beam receives 5.161×10⁻⁵ instead of zero. Dipole wakes, with probe factor 1, pass the translation controls.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _elegant.py](../helpers/_elegant.py) · [View source](../helpers/index.md#_elegant)
- [Download _trwake.py](../helpers/_trwake.py) · [View source](../helpers/index.md#_trwake)

??? example "Reproducer: COL-23.py"

    [Download COL-23.py](COL-23.py)

    ```python
    """COL-23: TRWAKE quadrupole probe coordinates omit element misalignment

    TRWAKE quadrupole wakes (probe exponents 1, drive exponents 0) use each particle's global transverse
    coordinate even when the structure has misalignments DX and DY. The kick must use displacement from
    the structure, so translating beam and structure together must preserve the kicks, and a beam on the
    structure's axis must receive none. For a 0.6000 nC electron bunch represented by 18 macroparticles,
    a common shift of (3.100, −1.700) mm raises the trailing particle's horizontal slope kick from
    1.498×10⁻⁵ to 6.660×10⁻⁵, and an on-axis beam receives 5.161×10⁻⁵ instead of zero. Dipole wakes,
    with probe factor 1, pass the translation controls.

    The script tracks the same 18 electron macroparticles through a TRWAKE element with a damped-sine
    wake on a 2 ps grid, for the quadrupole wake centered and translated, a beam on the translated
    axis, and the dipole wake translated as the control. It compares every slope with a direct causal
    pair sum over the structure-relative coordinates, and shows that evaluating the probe at the global
    coordinate reproduces the observed slopes.
    """

    import sys

    import numpy as np
    from _elegant import modes, verdict
    from _trwake import ATOL, RTOL, N, beam, compare, pair_sum, track

    # Six arrival-time groups of three particles; offsets and momenta vary across the bunch.
    TIMES = np.repeat(np.arange(-25, 26, 10), 3) * 1e-12  # s
    RELATIVE = np.column_stack([np.linspace(-0.8, 0.9, N), np.linspace(0.5, -0.7, N)]) * 1e-3  # m
    MOMENTUM = np.linspace(31, 49, N)  # m_e c
    SHIFT = (3.1e-3, -1.7e-3)  # m, common translation of beam and structure

    # (label, shift, quadrupole wake, beam on the structure axis)
    CASES = (
        ("quadrupole, centered", (0.0, 0.0), True, False),
        ("quadrupole, translated (3.1, -1.7) mm", SHIFT, True, False),
        ("quadrupole, translated (-4.3, 2.6) mm", (-4.3e-3, 2.6e-3), True, False),
        ("quadrupole, beam on the translated axis", SHIFT, True, True),
        ("dipole control, translated (3.1, -1.7) mm", SHIFT, False, False),
    )

    defect = False
    for mode in modes():
        for label, shift, quadrupole, on_axis in CASES:
            position = np.broadcast_to(shift, (N, 2)).copy()
            if not on_axis:
                position += RELATIVE
            columns = beam(TIMES, position[:, 0], position[:, 1], MOMENTUM)
            drive = 0 if quadrupole else 1
            observed, _ = track(mode, columns, shift=shift, drive=drive)
            # Expected: the pair sum over coordinates relative to the structure.
            expected = pair_sum(columns, shift=shift, drive=drive)
            # Cause: the probe factor taken in the global frame instead.
            global_probe = pair_sum(columns, shift=shift, drive=drive, probe_offset=(0.0, 0.0))
            bad = compare(f"{mode} {label}", observed, expected)
            translated = shift != (0.0, 0.0)
            if translated and quadrupole:
                if bad:
                    # A mismatch must be explained by the global-frame probe to count as the defect.
                    if not np.allclose(observed, global_probe, rtol=RTOL, atol=ATOL):
                        sys.exit(f"{mode} {label}: kick follows neither the pair sum nor global probe")
                else:
                    print(f"{mode} {label}: kick agrees with the structure-relative pair sum")
                defect |= bad
            elif bad:
                sys.exit(
                    f"{mode} {label}: control disagrees with the pair sum; the comparison is invalid"
                )
    print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} on the slopes of all {N} particles")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | master (`75a2e1a1a58b`), source with SDDS a60b9bbed7cc, CPU, serial and 3-rank Pelegant |

