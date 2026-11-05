---
title: "OPT-25: srdt counts skew and rotated multipoles as normal ones"
hide:
  - toc
---

# OPT-25 — srdt counts skew and rotated multipoles as normal ones

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [make_slices, bmad/modules/srdt_mod.f90:547–597](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L547-L597)<br>[srdt_calc, bmad/modules/srdt_mod.f90:110–142](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/modules/srdt_mod.f90#L110-L142)<br>[Bmad manual, Transfer Maps: Driving Terms, bmad/doc/subroutines.tex:3476–3513](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/subroutines.tex#L3476-L3513) |

When srdt evaluates skew or rotated multipoles, it discards the multipole rotation and uses the returned strength as a normal component. Normal resonance-driving terms require the real part of the rotated multipole coefficient. In a FODO ring, a pure skew sextupole produces the normal sextupole driving coefficient h₂₁₀₀₀ = −1.865 instead of zero, and rotating a normal thin multipole by π/3 leaves −1.865 instead of reversing it to +1.865. A mixed sextupole with integrated normal and skew strengths 0.5 and 0.2 m⁻² is overcounted by 7.703%.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _srdt_wiggler_maps.py](../helpers/_srdt_wiggler_maps.py) · [View source](../helpers/index.md#_srdt_wiggler_maps)

??? example "Reproducer: OPT-25.py"

    [Download OPT-25.py](OPT-25.py)

    ```python
    """OPT-25: srdt counts skew and rotated multipoles as normal ones

    When srdt evaluates skew or rotated multipoles, it discards the multipole rotation and uses the returned strength as a normal component. Normal resonance-driving terms require the real part of the rotated multipole coefficient. In a FODO ring, a pure skew sextupole produces the normal sextupole driving coefficient h₂₁₀₀₀ = −1.865 instead of zero, and rotating a normal thin multipole by π/3 leaves −1.865 instead of reversing it to +1.865. A mixed sextupole with integrated normal and skew strengths 0.5 and 0.2 m⁻² is overcounted by 7.703%.

    The script puts a thin element at zero betatron phase in a 16-bend FODO ring at 1 GeV total electron energy. It compares normal, skew, tilted, pole-rotated, reversed and mixed sextupoles with an independent expansion of their kick potential. BMAD_DIST selects the production build.
    """

    import numpy as np
    from _bmad import compare, verdict
    from _srdt_wiggler_maps import srdt_case

    cases = [
        ("normal", "multipole, k2l=1", 0.5),
        ("skew", "multipole, k2sl=1", 0),
        ("tilted", "multipole, k2l=1, tilt=pi/6", 0),
        ("pole rotation", "multipole, k2l=1, t2=-pi/6", 0),
        ("reversed", "multipole, k2l=1, tilt=pi/3", -0.5),
        ("mixed", "multipole, k2l=.5, k2sl=.2", 0.25),
    ]
    _, base = srdt_case("multipole, k2l=0")
    defect = False
    for name, spec, b2 in cases:
        twiss, actual = srdt_case(spec)
        ba, bb, eta, pa, pb = twiss
        np.testing.assert_allclose([pa, pb], 0, atol=1e-14)
        # The normal sextupole potential is b2*(x^3/3-x*y^2); a skew potential
        # contains different monomials. Expanding x=sqrt(beta_x)*(h+ + h-)/2
        # gives these first-order coefficients at zero betatron phase.
        expected = np.column_stack(
            (
                [-b2 * ba**1.5 / 8, b2 * ba**0.5 * bb / 4, -b2 * eta * ba / 2, b2 * eta * bb / 2],
                np.zeros(4),
            )
        )
        actual[2:] -= base[2:]
        bad = compare(
            f"{name}: h21000,h10110,delta_h11001,delta_h00111 (real,imag)", actual, expected, atol=2e-12
        )
        if name == "normal" and bad:
            raise AssertionError("Normal sextupole control failed")
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found (source lead for skew/rotated strengths in srdt) | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`) |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-24 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |
| Confirmed (thin skew and rotated sextupole terms by independent analytic open-line test) | `gpt-6-astra` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

