---
title: "XLT-07: MAD-X bend import treats total k0 as a field error"
hide:
  - toc
---

# XLT-07 — MAD-X bend import treats total k0 as a field error

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P1 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [`parse_and_write_element`, util_programs/mad_to_bmad/madx_to_bmad.py:548–555](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L548-L555)<br>[bmad/doc/elements.tex:514–516](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L514-L516)<br>[MAD-X bend.html:135–139](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/usrguide/Introduction/bend.html#L135-L139) |

Importing a MAD-X sector bend with explicit nonzero total dipole strength k₀ assigns Bmad's field error dg = k₀, adding the design curvature g = angle/L to the intended field. For an ideal 1 m bend with angle 0.1 rad and k₀ = 0.1 m⁻¹, this doubles the field to 0.2 m⁻¹. An on-axis particle then exits at x = −49.96 mm and normalized horizontal momentum pₓ = −0.09983, matching an independent circle calculation, while MAD-X keeps it on the design orbit. The correct import without taper is dg = k₀ − g, zero here. The same sector bend without explicit k₀ imports correctly.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _translators.py](../helpers/_translators.py) · [View source](../helpers/index.md#_translators)

??? example "Reproducer: XLT-07.py"

    [Download XLT-07.py](XLT-07.py)

    ```python
    """XLT-07: MAD-X bend import treats total k0 as a field error

    Importing a MAD-X sector bend with explicit nonzero total dipole strength k₀ assigns Bmad's
    field error dg = k₀, adding the design curvature g = angle/L to the intended field. For an ideal
    1 m bend with angle 0.1 rad and k₀ = 0.1 m⁻¹, this doubles the field to 0.2 m⁻¹. An on-axis
    particle then exits at x = −49.96 mm and normalized horizontal momentum pₓ = −0.09983, matching
    an independent circle calculation, while MAD-X keeps it on the design orbit. The correct import
    without taper is dg = k₀ − g, zero here. The same sector bend without explicit k₀ imports
    correctly.

    The script translates with madx_to_bmad.py a MAD-X sbend with L = 1 m, angle 0.1 rad and
    k0 = 0.1 m⁻¹, the ideal bend written with its field explicit, and the same bend without k0 as
    the control. It reads the imported dg and tracks the imported lattice in Tao from a zero orbit.
    MAD-X TWISS of the source deck, which must stay on the design orbit, is the receiving-code
    reference, and the exact circle in the doubled field accounts for the observed orbit. Set BMAD_SOURCE to the pinned source checkout.
    """

    from _bmad import compare, verdict
    from _translators import bend_exit, import_madx, madx, numbers, tao, value

    G, LENGTH = 0.1, 1.0
    SOURCE = (
        "beam, particle = electron, energy = 3;\n"
        f"b: sbend, l = {LENGTH}, angle = {G * LENGTH}{{}};\n"
        "ll: line = (b);\nuse, period = ll;\n"
    )
    OPEN = "parameter[geometry] = open\nbeginning[beta_a] = 10\nbeginning[beta_b] = 10\n"


    def imported_orbit(deck):
        """Imported Bmad text and the Tao exit (x, px) of the imported bend from a zero orbit."""
        text = import_madx(deck)
        orbit = numbers(tao(text + OPEN, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["x", "px"])
        return text, orbit


    defect = False
    for k0 in (None, G):
        deck = SOURCE.format("" if k0 is None else f", k0 = {k0}")
        label = "no k0" if k0 is None else f"k0 = {k0}"
        with madx(deck) as mad:
            twiss = mad.twiss(betx=10, bety=10)
            madx_end = [twiss.x[-1], twiss.px[-1]]
        # k0 = h = angle/L is MAD-X's ideal bend: the orbit stays on the design circle.
        if compare(
            f"{label}: MAD-X TWISS exit (x m, px), expected design orbit", madx_end, [0, 0], atol=1e-15
        ):
            raise SystemExit("MAD-X leaves the design orbit: comparison not valid.")
        text, orbit = imported_orbit(deck)
        bend = next(s for s in text.splitlines() if s.startswith("b:"))
        print(f"{label}: imported bend: {bend}")
        observed_dg = value(bend, "dg", 0.0)
        bad = compare(f"{label}: imported dg (1/m), expected k0 - angle/L", observed_dg, 0, atol=1e-15)
        bad |= compare(
            f"{label}: Bmad exit (x m, px), expected design orbit", orbit, [0, 0], atol=1e-14
        )
        if k0 is None and bad:
            raise SystemExit("The bend without k0 is not imported as an ideal bend: not valid.")
        if k0 is not None:
            defect |= bad
            # The orbit is that of the exact circle in g + dg = 0.2 1/m (closed form, rounding 1e-15 m).
            doubled = bend_exit(G, G + observed_dg, G * LENGTH)
            if compare(
                f"{label}: Bmad exit, expected exact circle in g + dg", orbit, doubled, atol=1e-14
            ):
                raise SystemExit("The imported orbit is not the circle of g + dg: not valid.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

