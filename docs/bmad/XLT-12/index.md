---
title: "XLT-12: Ordinary MAD-X aperture import unpacks the wrong array slice"
hide:
  - toc
---

# XLT-12 — Ordinary MAD-X aperture import unpacks the wrong array slice

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Source | [parse_and_write_element aperture conversion, util_programs/mad_to_bmad/madx_to_bmad.py:640–663](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L640-L663) |

Importing a MAD-X ellipse or rectangle with two half-apertures crashes the translator. It always selects the third and fourth aperture values, leaving an empty slice for ordinary two-value shapes. An ellipse with half-apertures 0.01 m and 0.02 m raises ValueError instead of producing those Bmad limits. MAD-X accepts the input, and Bmad accepts the corresponding explicit collimator.

Run `python XLT-12.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-12.py"

    [Download XLT-12.py](XLT-12.py)

    ```python
    """XLT-12: Ordinary MAD-X aperture import unpacks the wrong array slice

    Importing a MAD-X ellipse or rectangle with two half-apertures crashes the translator. It always selects the third and fourth aperture values, leaving an empty slice for ordinary two-value shapes. An ellipse with half-apertures 0.01 m and 0.02 m raises ValueError instead of producing those Bmad limits. MAD-X accepts the input, and Bmad accepts the corresponding explicit collimator.

    The script invokes the importer from BMAD_SOURCE on ellipse and rectangle apertures accepted by MAD-X. It requires the specific unpacking error and checks equivalent Bmad collimators as controls.
    """

    from _bmad import verdict
    from _lattice_import_export import HEADER, inspect, madx, translate, valid

    defect = False
    for shape, kind in [("ellipse", "ecollimator"), ("rectangle", "rcollimator")]:
        lattice = f"beam,particle=electron,pc=3; c:collimator,l=1,apertype={shape},aperture={{.01,.02}}; ll:line=(c); use,period=ll;"
        with madx() as mad:
            mad.input(lattice)
            assert list(mad.elements.c.aperture) == [0.01, 0.02]
        valid(inspect(HEADER + f"c:{kind},l=1,x_limit=.01,y_limit=.02\nll:line=(c)\nuse,ll\n"))
        status, log, output = translate(lattice)
        bad = status == 1 and "ValueError: not enough values to unpack (expected 2, got 0)" in log
        if status and not bad:
            raise RuntimeError(log)
        if not bad:
            valid(inspect(output))
        print(
            f"{shape}: unpacking ValueError={bad}; expected limits=[0.01,0.02] m; MAD-X and Bmad controls accepted"
        )
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |

