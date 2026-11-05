---
title: "XLT-01: Converter angular cuts increase yield through an inverted acceptance factor"
hide:
  - toc
---

# XLT-01 — Converter angular cuts increase yield through an inverted acceptance factor

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [`probability_tables_setup`, track_a_converter.f90:421–428](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_converter.f90#L421-L428)<br>[Bmad manual, “Converter Tracking”, `s:converter.track`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/charged-tracking.tex#L754-L764) |

Applying an angular cut to a converter increases its outgoing physical yield instead of reducing it. For a normalized synthetic distribution and angle_out_max = 0.15, the accepted fraction is 0.06149, but the returned weight is 16.26. The weight is the reciprocal of the required fraction, so even a restrictive cut produces more weight than the unrestricted converter. Integrating the supplied angular density analytically establishes the required normalization.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_24_converter.py](../helpers/_s3_24_converter.py) · [View source](../helpers/index.md#_s3_24_converter)

??? example "Reproducer: XLT-01.py"

    [Download XLT-01.py](XLT-01.py)

    ```python
    """XLT-01: Converter angular cuts increase yield through an inverted acceptance factor

    Applying an angular cut to a converter increases its outgoing physical yield instead of reducing it. For a normalized synthetic distribution and angle_out_max = 0.15, the accepted fraction is 0.06149, but the returned weight is 16.26. The weight is the reciprocal of the required fraction, so even a restrictive cut produces more weight than the unrestricted converter. Integrating the supplied angular density analytically establishes the required normalization.

    The script tracks a fixed electron through supplied converter tables with and without the angular cut. It compares the weight with an analytic accepted fraction and checks that the unrestricted weight is one.
    """

    import math

    from _bmad import compare, verdict
    from _s3_24_converter import samples

    base, _, _ = samples()
    if compare("unrestricted weight, unit-integral table", base[0, 1], 1, atol=2e-14):
        raise RuntimeError("Unrestricted normalization control failed")
    cut, _, _ = samples("cv[angle_out_max]=0.15\n")
    # The manual and implementation use h=atan(angle_out_max). The x density is
    # constant; the y integral of 1/(1+4*y*y) is atan(2*y)/2.
    h = math.atan(0.15)
    fraction = 2 * h * math.atan(2 * h) / (1.6 * math.atan(1.2))
    verdict(
        compare("angular-cut weight, integral of accepted density", cut[0, 1], fraction, atol=2e-13)
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

