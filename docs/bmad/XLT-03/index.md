---
title: "XLT-03: Converter beta changes rescale longitudinal position backwards"
hide:
  - toc
---

# XLT-03 — Converter beta changes rescale longitudinal position backwards

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [`track_a_converter`, track_a_converter.f90:240–252](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/track_a_converter.f90#L240-L252)<br>[Bmad manual, “Charged Particle Phase Space Coordinates”, `s:phase.space`, equation `zbctt`](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/phase-space.tex#L81-L94) |

Converting a particle to a different speed rescales its longitudinal coordinate z by the inverse of the required speed ratio. The coordinate definition z = −βc(t − t₀), where β is speed divided by c and t₀ is reference time, requires z to scale as β_new/β_old when stored time stays fixed. In a 4 MeV/c electron-to-positron example, the returned z is 31.63 mm instead of 16.72 mm. The absolute clock and the post-drift speed-ratio calculation give the same expected value.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_24_converter.py](../helpers/_s3_24_converter.py) · [View source](../helpers/index.md#_s3_24_converter)

??? example "Reproducer: XLT-03.py"

    [Download XLT-03.py](XLT-03.py)

    ```python
    """XLT-03: Converter beta changes rescale longitudinal position backwards

    Converting a particle to a different speed rescales its longitudinal coordinate z by the inverse of the required speed ratio. The coordinate definition z = −βc(t − t₀), where β is speed divided by c and t₀ is reference time, requires z to scale as β_new/β_old when stored time stays fixed. In a 4 MeV/c electron-to-positron example, the returned z is 31.63 mm instead of 16.72 mm. The absolute clock and the post-drift speed-ratio calculation give the same expected value.

    The script compares the first fixed-seed conversion with a drift-only clock control. It checks the unchanged time and computes z both from the absolute clock and from the required speed ratio.
    """

    from _bmad import compare, verdict
    from _s3_24_converter import samples

    _, clock, absolute = samples()
    oldbeta, newbeta, oldtime, newtime, oldz, actual, expected, _ = clock
    if compare("stored time through conversion (s)", newtime, oldtime, atol=2e-24):
        raise RuntimeError("Conversion changed the clock")
    # z=-beta*c*(t-t0); changing beta at fixed time scales z by beta_new/beta_old.
    if compare(
        "absolute clock versus speed-ratio oracle (m)",
        absolute[1],
        oldz * newbeta / oldbeta,
        atol=2e-17,
    ):
        raise RuntimeError("Independent clock controls disagree")
    verdict(compare("converted z (m), fixed absolute time", actual, expected, atol=2e-17))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

