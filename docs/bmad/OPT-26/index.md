---
title: "OPT-26: srdt returns NaN for zero-length non-multipole elements with multipoles"
hide:
  - toc
---

# OPT-26 — srdt returns NaN for zero-length non-multipole elements with multipoles

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [make_slices, bmad/modules/srdt_mod.f90:558–597](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L558-L597)<br>[Bmad manual, Transfer Maps: Driving Terms, bmad/doc/subroutines.tex:3476–3513](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/subroutines.tex#L3476-L3513) |

Calling srdt on a zero-length ab_multipole or sad_mult with a nonzero sextupole component returns NaN driving terms. Only the multipole element class takes the thin-element path; the other classes divide integrated strength by zero length and then multiply by a zero slice length. Equivalent thin elements should return the same finite terms. In a FODO ring, both affected representations return NaN where the equivalent multipole gives h₂₁₀₀₀ = −1.865 and h₁₀₁₁₀ = 0.9849.

Run `python OPT-26.py` with [_bmad.py](../helpers/_bmad.py) and [_srdt_wiggler_maps.py](../helpers/_srdt_wiggler_maps.py) in the same directory.

??? example "Reproducer: OPT-26.py"

    [Download OPT-26.py](OPT-26.py)

    ```python
    """OPT-26: srdt returns NaN for zero-length non-multipole elements with multipoles

    Calling srdt on a zero-length ab_multipole or sad_mult with a nonzero sextupole component returns NaN driving terms. Only the multipole element class takes the thin-element path; the other classes divide integrated strength by zero length and then multiply by a zero slice length. Equivalent thin elements should return the same finite terms. In a FODO ring, both affected representations return NaN where the equivalent multipole gives h₂₁₀₀₀ = −1.865 and h₁₀₁₁₀ = 0.9849.

    The script places equivalent multipole, ab_multipole and sad_mult sextupoles at zero phase in the same FODO ring. It validates the finite multipole control against the analytic normal-sextupole coefficients before checking the two other representations for NaNs.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _srdt_wiggler_maps import srdt_case

    twiss, control = srdt_case("multipole, k2l=1")
    ba, bb = twiss[:2]
    expected = np.array([[-(ba**1.5) / 16, 0], [ba**0.5 * bb / 8, 0]])
    if compare("thin normal control: h21000,h10110", control[:2], expected, atol=2e-12):
        raise AssertionError("Thin multipole control failed")
    defect = False
    for spec in ("ab_multipole, b2=.5", "sad_mult, b2=.5"):
        _, actual = srdt_case(spec)
        if np.isnan(actual).any():
            print(
                f"{spec}: observed={actual[:2].tolist()} expected={expected.tolist()} from the identical thin normal potential"
            )
            defect = True
        else:
            defect |= compare(spec, actual[:2], expected, atol=2e-12)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed<br><small>ab_multipole and sad_mult thin-sextupole NaNs</small> | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

