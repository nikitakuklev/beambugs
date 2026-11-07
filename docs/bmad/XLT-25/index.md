---
title: "XLT-25: Elegant export of a zero-length magnet writes NaN"
hide:
  - toc
---

# XLT-25 — Elegant export of a zero-length magnet writes NaN

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">experimental</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [bend integrated-strength division, bmad/output/write_lattice_elegant_format.f90:581–586](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L581-L586)<br>[quadrupole, sextupole and octupole divisions, bmad/output/write_lattice_elegant_format.f90:600–637](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L600-L637)<br>[multipole scaling, bmad/code/multipole_ele_to_ab.f90:535–572](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/code/multipole_ele_to_ab.f90#L535-L572) |

Exporting a zero-length quadrupole, sextupole, octupole, or zero-curvature bend to Elegant divides integrated strengths by zero. The output contains NaN strength fields even though the Bmad lattice is valid and the zero-length body has no integrated field. A quadrupole with normalized gradient k₁ = 1.2 m⁻² exports NaN k₁ and sextupole strength k₂ instead of a finite identity body. A zero-length solenoid remains finite because its writer uses the strength directly.

Run `python XLT-25.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-25.py"

    [Download XLT-25.py](XLT-25.py)

    ```python
    """XLT-25: Elegant export of a zero-length magnet writes NaN

    Exporting a zero-length quadrupole, sextupole, octupole, or zero-curvature bend to Elegant divides integrated strengths by zero. The output contains NaN strength fields even though the Bmad lattice is valid and the zero-length body has no integrated field. A quadrupole with normalized gradient k₁ = 1.2 m⁻² exports NaN k₁ and sextupole strength k₂ instead of a finite identity body. A zero-length solenoid remains finite because its writer uses the strength directly.

    The script exports zero-length magnets and a solenoid control to Elegant, and checks their corresponding MAD-X exports for finite fields. It also exports finite-length versions to distinguish the zero-length division from ordinary strength conversion.
    """

    import re

    from _bmad import verdict
    from _lattice_import_export import HEADER, definition, export

    body = """q:quadrupole,l=LENGTH,k1=1.2
    sx:sextupole,l=LENGTH,k2=2
    o:octupole,l=LENGTH,k3=3
    b:sbend,l=LENGTH,angle=0
    so:solenoid,l=LENGTH,ks=.2
    lat:line=(q,sx,o,b,so)
    use,lat
    """
    finite = export(HEADER + body.replace("LENGTH", ".5"), "elegant")
    if re.search(r"\b(?:nan|inf)\b", finite, re.IGNORECASE):
        raise RuntimeError(finite)
    zero = HEADER + body.replace("LENGTH", "0")
    madx = export(zero)
    if re.search(r"\b(?:nan|inf)\b", madx, re.IGNORECASE):
        raise RuntimeError(madx)
    text = export(zero, "elegant")
    if re.search(r"\b(?:nan|inf)\b", definition(text, "SO"), re.IGNORECASE):
        raise RuntimeError(text)
    print(
        "controls: finite-length Elegant, zero-length MAD-X, and zero-length Elegant solenoid fields are finite"
    )
    defect = False
    for name in ["Q", "SX", "O", "B"]:
        row = definition(text, name)
        bad = bool(re.search(r"\bnan\b", row, re.IGNORECASE))
        print(f"{name}: {row}; expected finite identity-body representation")
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-21 | foreign-export-embedded-kicks (`5afc3c33a462`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-22 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

