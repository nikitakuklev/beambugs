---
title: "XLT-06: MAD-X bend export omits the design field and doubles its error"
hide:
  - toc
---

# XLT-06 — MAD-X bend export omits the design field and doubles its error

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_mad_format`, bmad/output/write_lattice_mad_format.f90:772–776](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L772-L776)<br>[`multipole_ele_to_ab`, bmad/code/multipole_ele_to_ab.f90:379](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_ab.f90#L379)<br>[bmad/doc/elements.tex:514–516](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L514-L516)<br>[MAD-X bend.html:135–139](https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/7998e738eca575c82a424138517c2a17115f3cdb/doc/usrguide/Introduction/bend.html#L135-L139) |

Exporting an sbend with a field error dg or horizontal steering to MAD-X writes k₀ = dg + b₀/L, omitting the design curvature g and counting dg twice through the integrated dipole coefficient b₀. MAD-X requires the total dipole strength, g + b₀/L, where L is the bend length. For L = 1 m, angle 0.1 rad and dg = 0.001 m⁻¹, the export writes 0.002 m⁻¹ instead of 0.101 m⁻¹; the exit position is 49.08 mm instead of the exact-circle −0.4996 mm that Bmad gives. With horizontal kick 0.1 mrad, it is 50.13 mm instead of 0.04996 mm. Ideal bends written without k₀ are unaffected.

Run `python XLT-06.py` with [_bmad.py](../helpers/_bmad.py) and [_translators.py](../helpers/_translators.py) in the same directory.

??? example "Reproducer: XLT-06.py"

    [Download XLT-06.py](XLT-06.py)

    ```python
    """XLT-06: MAD-X bend export omits the design field and doubles its error

    Exporting an sbend with a field error dg or horizontal steering to MAD-X writes k₀ = dg + b₀/L,
    omitting the design curvature g and counting dg twice through the integrated dipole coefficient
    b₀. MAD-X requires the total dipole strength, g + b₀/L, where L is the bend length. For L = 1 m,
    angle 0.1 rad and dg = 0.001 m⁻¹, the export writes 0.002 m⁻¹ instead of 0.101 m⁻¹; the exit
    position is 49.08 mm instead of the exact-circle −0.4996 mm that Bmad gives. With horizontal
    kick 0.1 mrad, it is 50.13 mm instead of 0.04996 mm. Ideal bends written without k₀ are
    unaffected.

    The script exports with bmad_to_mad_sad_elegant an sbend with L = 1 m and angle 0.1 rad that
    carries either a field error dg = 0.001 m⁻¹ or an embedded steering kick hkick = 0.1 mrad,
    reads the written k0, and runs MAD-X TWISS on the export from a zero orbit. Both are compared
    with the total field g + dg − hkick/L and with the exit orbit of the exact circle in that field.
    Bmad's own exit orbit is the control, and a copy of the export with only k0 set to the total
    field is the receiving-code control.
    """

    import re

    from _bmad import compare, verdict
    from _translators import bend_exit, export, lattice, madx, numbers, statements, tao, value

    G, LENGTH = 0.1, 1.0
    # MAD-X TWISS propagates the orbit through a bend with k0 != h by a second-order map; with the
    # correct k0 its truncation leaves 1.2e-10 m in x, so the MAD-X comparisons use 1e-9.
    MADX_ATOL = 1e-9


    def madx_orbit(text):
        with madx(text) as mad:
            twiss = mad.twiss(beta0="initial")
            return [twiss.x[-1], twiss.px[-1]]


    defect = False
    for label, attribute, total in (
        ("dg = 0.001", "dg = 0.001", G + 0.001),
        # A bend's hkick is a uniform field change toward +x: total field g - hkick/L.
        ("hkick = 1e-4", "hkick = 1e-4", G - 1e-4 / LENGTH),
    ):
        # Exact on-axis exit orbit in the total field, relative to the design circle of g.
        expected = bend_exit(G, total, G * LENGTH)
        lat = lattice(
            f"b: sbend, l = {LENGTH}, angle = {G * LENGTH}, {attribute}, fringe_type = none", "b"
        )
        bmad = numbers(tao(lat, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["x", "px"])
        # The closed form subtracts coordinates of size 1/g = 10 m: rounding is about 1e-15 m.
        if compare(f"{label}: Bmad exit (x m, px), expected exact circle", bmad, expected, atol=1e-14):
            raise SystemExit("Bmad disagrees with the exact circle: comparison not valid.")
        deck = export(lat, "madx")
        k0 = value(statements(deck, "madx")["B"], "k0")
        defect |= compare(f"{label}: exported k0 (1/m), expected total field", k0, total, atol=1e-15)
        defect |= compare(
            f"{label}: MAD-X TWISS exit (x m, px) of the export, expected exact circle",
            madx_orbit(deck),
            expected,
            atol=MADX_ATOL,
        )
        fixed = re.sub(r"\bk0 =\s*[^,;\s]+", f"k0 = {total!r}", deck, count=1)
        if compare(
            f"{label}: MAD-X TWISS exit with only k0 set to the total field, expected exact circle",
            madx_orbit(fixed),
            expected,
            atol=MADX_ATOL,
        ):
            raise SystemExit("The total-field k0 control disagrees: comparison not valid.")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

