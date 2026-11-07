---
title: "XLT-10: Elegant collimator export reads another element's length"
hide:
  - toc
---

# XLT-10 — Elegant collimator export reads another element's length

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`write_lattice_elegant_format`, bmad/output/write_lattice_elegant_format.f90:700–706](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L700-L706)<br>[line 681](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L681)<br>[bmad/doc/elements.tex:3851–3854](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/elements.tex#L3851-L3854)<br>[line 120](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L120) |

Exporting a Bmad rcollimator or ecollimator to Elegant writes a scratch multipole element's length instead of the collimator's length. With no multipoles, a 2 m rectangular collimator followed by a 1 m drift becomes a 1 m line. A particle with slope 1 mrad exits at longitudinal position s = 1.000 m and x = 1.000 mm instead of s = 3.000 m and x = 3.000 mm, as Bmad and straight-line propagation give. After a multipole with fictitious length 0.5 m, a 2 m elliptical collimator is written with l = 0.5 m. Using the collimator's own length restores the rectangular case's trajectory.

Run `python XLT-10.py` with [_bmad.py](../helpers/_bmad.py) and [_translators.py](../helpers/_translators.py) in the same directory.

??? example "Reproducer: XLT-10.py"

    [Download XLT-10.py](XLT-10.py)

    ```python
    """XLT-10: Elegant collimator export reads another element's length

    Exporting a Bmad rcollimator or ecollimator to Elegant writes a scratch multipole element's
    length instead of the collimator's length. With no multipoles, a 2 m rectangular collimator
    followed by a 1 m drift becomes a 1 m line. A particle with slope 1 mrad exits at longitudinal
    position s = 1.000 m and x = 1.000 mm instead of s = 3.000 m and x = 3.000 mm, as Bmad and
    straight-line propagation give. After a multipole with fictitious length 0.5 m, a 2 m elliptical
    collimator is written with l = 0.5 m. Using the collimator's own length restores the rectangular
    case's trajectory.

    Setup: serial Elegant on `PATH` and `RPN_DEFNS` set to the `defns.rpn` file of SDDS; the script
    tracks the exported lattice in Elegant.

    The script exports with bmad_to_mad_sad_elegant a 2 m rectangular collimator followed by a 1 m
    drift, tracks a particle with slope 1 mrad through the export in Elegant, and compares the exit
    transverse and longitudinal positions with Bmad's and with straight-line propagation. A copy of the
    export with only `l = 2` added to the collimator is the receiving-code control. A second case puts a
    multipole with the fictitious length 0.5 m before a 2 m elliptical collimator and reads the length
    written for the collimator.
    """

    import re

    from _bmad import compare, verdict
    from _translators import elegant_track, export, lattice, numbers, statements, tao, value

    P0C, XP = 1e9, 1e-3
    APERTURE = "x_limit = 0.01, y_limit = 0.02"
    # A field-free 2 m collimator and 1 m drift: s = 3 m and x = 3 m * x' (x' = 1 mrad).
    EXPECTED = [3.0, 3.0 * XP]
    # Elegant starts from the slope x'; Bmad from the canonical px = x'/sqrt(1 + x'^2) at p = p0.
    PX = XP / (1 + XP**2) ** 0.5

    lat = lattice(f"c: rcollimator, l = 2, {APERTURE}\nd: drift, l = 1", "c, d", p0c=P0C, start=[0, PX])
    bmad = numbers(tao(lat, ["pipe ele:orbit 2"])["pipe ele:orbit 2"], ["s", "x"])
    if compare("Bmad exit (s m, x m)", bmad, EXPECTED, atol=1e-15):
        raise SystemExit("Bmad disagrees with straight-line propagation: comparison not valid.")

    lte = export(lat, "elegant")
    print(f"exported: C: {statements(lte, 'elegant')['C']}")
    end = elegant_track(lte, P0C / 1e6, [0, XP, 0, 0])
    defect = compare("Elegant exit (s m, x m) of the export", end[:2], EXPECTED, atol=1e-15)
    fixed, count = re.subn(r"^C: rcol,", "C: rcol, l = 2,", lte, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit("Expected one rcol definition for C.")
    end = elegant_track(fixed, P0C / 1e6, [0, XP, 0, 0])
    if compare("Elegant exit (s m, x m) with l = 2 added", end[:2], EXPECTED, atol=1e-15):
        raise SystemExit("The l = 2 control disagrees: comparison not valid.")

    # The multipole's length "affects the longitudinal position of the next element but does not
    # affect any tracking" (Bmad manual); the collimator's own length is 2 m.
    stale = lattice(
        f"m: multipole, l = 0.5, k2l = 0.1\nc: ecollimator, l = 2, {APERTURE}", "m, c", p0c=P0C
    )
    written = statements(export(stale, "elegant"), "elegant")["C"]
    print(f"after a multipole with l = 0.5: C: {written}")
    observed = value(written, "l", 0.0)
    defect |= compare("ecol after the multipole: written l (m), expected 2", observed, 2.0, atol=0)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

