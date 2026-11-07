---
title: "XLT-08: SAD MULT fallback invents multipoles through order 21"
hide:
  - toc
---

# XLT-08 — SAD MULT fallback invents multipoles through order 21

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [write_lattice_sad_format, bmad/output/write_lattice_sad_format.f90:703–738](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_sad_format.f90#L703-L738)<br>[multipole1_kt_to_ab, bmad/low_level/multipole1_kt_to_ab.f90:18–39](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/low_level/multipole1_kt_to_ab.f90#L18-L39) |

A sextupole with an embedded horizontal kick takes the SAD MULT export path, which loses its body strength and creates normal and skew multipoles at unrelated orders. A 0.5 m sextupole with k₂ = 2 m⁻³ and a 0.1 mrad horizontal kick exports K₂ = −0.0002000 instead of 1.000 m⁻², SK₀ = 1.571 instead of zero, and SK₂₁ = 8.025 × 10¹⁹ instead of zero. The same sextupole without the kick exports K₂ correctly. The scalar-to-array additions and misplaced rotation arguments explain the spurious coefficients.

Run `python XLT-08.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-08.py"

    [Download XLT-08.py](XLT-08.py)

    ```python
    """XLT-08: SAD MULT fallback invents multipoles through order 21

    A sextupole with an embedded horizontal kick takes the SAD MULT export path, which loses its body strength and creates normal and skew multipoles at unrelated orders. A 0.5 m sextupole with k₂ = 2 m⁻³ and a 0.1 mrad horizontal kick exports K₂ = −0.0002000 instead of 1.000 m⁻², SK₀ = 1.571 instead of zero, and SK₂₁ = 8.025 × 10¹⁹ instead of zero. The same sextupole without the kick exports K₂ correctly. The scalar-to-array additions and misplaced rotation arguments explain the spurious coefficients.

    The script exports a plain sextupole and a sextupole with an embedded kick to SAD. It compares the exported integrated multipoles with the body strength times length and the signed horizontal kick.
    """

    import re

    from _bmad import compare, verdict
    from _lattice_import_export import HEADER, export, number

    text = export(
        HEADER
        + """s0: sextupole, l=.5, k2=2
    s1: sextupole, l=.5, k2=2, hkick=.0001
    lat: line=(s0,s1)
    use,lat
    """,
        "sad",
    )


    def coefficients(name):
        match = re.search(rf"\b{name}\s*=\s*\((.*?)\)", text, re.DOTALL | re.IGNORECASE)
        if match is None:
            raise RuntimeError(text)
        return match[1]


    if compare("plain sextupole K2 (m^-2), k2*L", number(coefficients("S0"), "K2"), 1, atol=1e-15):
        raise RuntimeError("Plain sextupole control failed")
    case = coefficients("S1")
    # SAD K_n is the integrated normal strength, and K0=-hkick in this frame.
    # The input contains no skew field or higher-order field.
    keys = ["K0", "K2", "SK0", "SK21"]
    # Omitted SAD coefficients are zero by definition.
    values = [number(case, k) if re.search(rf"\b{k}\s*=", case, re.IGNORECASE) else 0.0 for k in keys]
    verdict(compare("kicked sextupole [K0,K2,SK0,SK21]", values, [-1e-4, 1, 0, 0], atol=2e-15))

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

