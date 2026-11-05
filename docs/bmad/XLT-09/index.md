---
title: "XLT-09: Elegant export writes quadratic Taylor terms as linear entries"
hide:
  - toc
---

# XLT-09 — Elegant export writes quadratic Taylor terms as linear entries

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_elegant_format`, bmad/output/write_lattice_elegant_format.f90:646–661](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L646-L661)<br>[rtsoliday/elegant `0355dd6aca12`, src/track_data.c:3011 and 3076](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/track_data.c#L3011-L3076)<br>[line 120](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L120) |

Exporting a Bmad taylor element to Elegant writes second-order coefficients as first-order matrix entries, applying quadratic terms as linear focusing and overwriting linear coefficients with the same name. For normalized momentum pₓ and the map pₓ → pₓ − 2x², a particle at x = 1 mm leaves Elegant with slope x′ = −2.000 mrad instead of −2.000 µrad, as the polynomial and Bmad give. Adding the linear term −0.5x leaves the result at −2.000 mrad instead of −0.5020 mrad. Writing the coefficient as t211 with order = 2 restores the expected slopes within the canonical-momentum to slope conversion error.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _translators.py](../helpers/_translators.py) · [View source](../helpers/index.md#_translators)

??? example "Reproducer: XLT-09.py"

    [Download XLT-09.py](XLT-09.py)

    ```python
    """XLT-09: Elegant export writes quadratic Taylor terms as linear entries

    Exporting a Bmad taylor element to Elegant writes second-order coefficients as first-order
    matrix entries, applying quadratic terms as linear focusing and overwriting linear coefficients
    with the same name. For normalized momentum pₓ and the map pₓ → pₓ − 2x², a particle at x = 1 mm
    leaves Elegant with slope x′ = −2.000 mrad instead of −2.000 µrad, as the polynomial and Bmad
    give. Adding the linear term −0.5x leaves the result at −2.000 mrad instead of −0.5020 mrad.
    Writing the coefficient as t211 with order = 2 restores the expected slopes within the
    canonical-momentum to slope conversion error.

    The script exports with bmad_to_mad_sad_elegant two Bmad taylor maps, the identity plus the
    quadratic term px += −2 x², and the same map with a linear term px += −0.5 x added. It tracks
    a particle at x = 1 mm through each export in Elegant and compares the exit slope with the
    polynomial value and with Bmad. An independently written EMATRIX with `t211` and `order = 2`
    is the receiving-code control; Elegant must be on PATH with RPN_DEFNS set.
    """

    from _bmad import compare, verdict
    from _translators import elegant_track, export, lattice, numbers, statements, tao

    P0C, X0 = 1e9, 1e-3
    IDENTITY = "{1: 1|1}, {2: 1|2}, {3: 1|3}, {4: 1|4}, {5: 1|5}, {6: 1|6}"
    # Elegant tracks the slope x' = px/sqrt((1 + delta)^2 - px^2 - py^2), equal to px to px^3/2,
    # below 1e-10 here; Bmad evaluates the polynomial in px exactly.
    ATOL = 1e-10

    defect = False
    for linear in (0.0, -0.5):
        terms = IDENTITY.replace("{2: 1|2}", "{2: 1|2}" + (f", {{2: {linear}|1}}" if linear else ""))
        lat = lattice(f"t: taylor, {terms}, {{2: -2|11}}", "t", p0c=P0C, start=[X0])
        expected = linear * X0 - 2 * X0**2  # px = px0 + linear x + (-2) x^2, px0 = 0
        label = f"linear term {linear:g}"
        bmad = numbers(tao(lat, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["px"])[0]
        if compare(f"{label}: Bmad exit px, expected polynomial", bmad, expected, atol=1e-18):
            raise SystemExit("Bmad disagrees with the map polynomial: comparison not valid.")
        lte = export(lat, "elegant")
        print(f"{label}: exported: T: {statements(lte, 'elegant')['T']}")
        end = elegant_track(lte, P0C / 1e6, [X0, 0, 0, 0])
        defect |= compare(f"{label}: Elegant exit xp, expected polynomial", end[2], expected, atol=ATOL)
        # An independent receiving-code control, derived from the same polynomial. It remains
        # valid if the exporter is fixed or changes its coefficient formatting.
        fixed = (
            f"T: ematrix, order=2, r11=1, r22=1, r33=1, r44=1, r55=1, r66=1, r21={linear}, t211=-2\n"
            "lat: line=(T)\n"
        )
        end = elegant_track(fixed, P0C / 1e6, [X0, 0, 0, 0])
        if compare(
            f"{label}: Elegant exit xp with t211, expected polynomial", end[2], expected, atol=ATOL
        ):
            raise SystemExit("The t211 control disagrees: comparison not valid.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

