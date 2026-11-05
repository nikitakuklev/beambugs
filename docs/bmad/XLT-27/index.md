---
title: "XLT-27: Generated element names can replace lattice elements of the same name"
hide:
  - toc
---

# XLT-27 — Generated element names can replace lattice elements of the same name

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [MAD helper kicker names, bmad/output/write_lattice_mad_format.f90:290–300](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L290-L300)<br>[MAD definition deduplication, bmad/output/write_lattice_mad_format.f90:620–624](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_mad_format.f90#L620-L624)<br>[Elegant helper kicker names, bmad/output/write_lattice_elegant_format.f90:288–295](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L288-L295)<br>[Elegant definition deduplication, bmad/output/write_lattice_elegant_format.f90:485–488](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/output/write_lattice_elegant_format.f90#L485-L488) |

Foreign lattice export can replace a real element when its name collides with an automatically generated helper element. A kicked quadrupole creates Q_KICKER_1; if a following 2 m drift already has that name, the writer defines only the kicker and uses it at all three occurrences. The exported line loses the drift and gains an extra kick. Renaming the input drift preserves its definition and length.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _lattice_import_export.py](../helpers/_lattice_import_export.py) · [View source](../helpers/index.md#_lattice_import_export)

??? example "Reproducer: XLT-27.py"

    [Download XLT-27.py](XLT-27.py)

    ```python
    """XLT-27: Generated element names can replace lattice elements of the same name

    Foreign lattice export can replace a real element when its name collides with an automatically generated helper element. A kicked quadrupole creates Q_KICKER_1; if a following 2 m drift already has that name, the writer defines only the kicker and uses it at all three occurrences. The exported line loses the drift and gains an extra kick. Renaming the input drift preserves its definition and length.

    The script exports the colliding and renamed lattices to MAD-X and Elegant. It compares the generated definitions and line references, requiring the renamed 2 m drift to survive in each format.
    """

    import re

    from _bmad import compare, verdict
    from _lattice_import_export import HEADER, definition, export, number

    body = """q:quadrupole,l=.5,k1=1.2,vkick=2e-5
    q_kicker_1:drift,l=2
    lat:line=(q,q_kicker_1)
    use,lat
    """
    defect = False
    for fmt in ["madx", "elegant"]:
        control = export(HEADER + body.replace("q_kicker_1", "user_drift"), fmt)
        row = definition(control, "USER_DRIFT")
        assert row.lower().startswith(("drift", "edrift"))
        if compare(f"{fmt}: renamed drift length control (m)", number(row, "l"), 2, atol=0):
            raise RuntimeError("Renamed drift control failed")
        text = export(HEADER + body, fmt)
        row = definition(text, "Q_KICKER_1")
        bad = not row.lower().startswith(("drift", "edrift"))
        # Three line references all resolve to the sole first definition.
        flat = re.sub(r"&\s*\n\s*", " ", text)
        line = next(s for s in flat.splitlines() if re.search(r"\bline\s*=", s, re.IGNORECASE))
        count = len(re.findall(r"\bQ_KICKER_1\b", line, re.IGNORECASE))
        if bad:
            if count != 3 or not row.lower().startswith(("kicker", "ekicker")):
                raise RuntimeError("Export changed beyond the isolated helper-name collision")
            if number(row, "vkick") != 1e-5:
                raise RuntimeError("Unexpected helper-kicker strength")
        elif number(row, "l") != 2:
            raise RuntimeError("Original drift has an unexpected length")
        print(
            f"{fmt}: Q_KICKER_1={row}; references={count}; expected original 2 m drift plus two distinct helper kicks"
        )
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6.1-sol` | xhigh | 2026-10-22 | foreign-export-embedded-kicks (`9da954b453cc`), source, production, CPU, double, serial |
| Confirmed | `claude-opus-5-5` | xhigh | 2026-10-22 | 20261021-0 (`d00a8f17e55a`), source, production, CPU, double, serial |

