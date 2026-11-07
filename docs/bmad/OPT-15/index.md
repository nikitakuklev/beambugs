---
title: "OPT-15: Periodic dispersion mixes normalized and raw matrix coordinates"
hide:
  - toc
---

# OPT-15 — Periodic dispersion mixes normalized and raw matrix coordinates

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [transverse normalization, bmad/code/twiss_from_mat6.f90:50–58](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/twiss_from_mat6.f90#L50-L58)<br>[periodic dispersion solve, bmad/code/twiss_from_mat6.f90:108–130](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/twiss_from_mat6.f90#L108-L130)<br>[Bmad manual, “Dispersion Calculation”, `s:dispersion`, `dexds`, bmad/doc/optics.tex:381–417](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/optics.tex#L381-L417) |

Enabling normalize_twiss in the periodic Twiss initializer at nonzero fractional momentum offset δ changes the physical dispersion by mixing normalized and ordinary coordinates. For the transverse one-turn matrix M₄ and momentum-source column c, the closed-orbit derivative D must solve (I − M₄)D = c in one coordinate system. A map at δ = 0.2 with D = (0.8, 0.13, 0.4, −0.05) returns (0.8951, 0.1170, 0.4223, −0.04347) when normalization is enabled, while disabling it recovers D.

Run `python OPT-15.py` with [_bmad.py](../helpers/_bmad.py) and [_dispersion_and_moments.py](../helpers/_dispersion_and_moments.py) in the same directory.

??? example "Reproducer: OPT-15.py"

    [Download OPT-15.py](OPT-15.py)

    ```python
    """OPT-15: Periodic dispersion mixes normalized and raw matrix coordinates

    Enabling normalize_twiss in the periodic Twiss initializer at nonzero fractional momentum
    offset δ changes the physical dispersion by mixing normalized and ordinary coordinates. For
    the transverse one-turn matrix M₄ and momentum-source column c, the closed-orbit derivative
    D must solve (I − M₄)D = c in one coordinate system. A map at δ = 0.2 with D = (0.8, 0.13,
    0.4, −0.05) returns (0.8951, 0.1170, 0.4223, −0.04347) when normalization is enabled, while
    disabling it recovers D.

    The script builds a stable periodic map whose fixed-point derivative is prescribed analytically. It
    executes the compiled initializer with normalization off and on, validates the first as a control,
    and compares the second with a raw-coordinate linear solve.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _dispersion_and_moments import periodic, require

    baseline, _, eta, _ = periodic(False)
    require(baseline[:4], eta)
    values, _, _, mat = periodic(True)
    # The dispersion is the derivative of the closed orbit fixed point.
    expected = np.linalg.solve(np.eye(4) - mat[:4, :4], mat[:4, 5])
    require(expected, eta)
    verdict(
        compare(
            "normalized periodic (eta_x, etap_x, eta_y, etap_y), fixed-point derivative",
            values[:4],
            expected,
            atol=1e-15,
            rtol=3e-14,
        )
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

