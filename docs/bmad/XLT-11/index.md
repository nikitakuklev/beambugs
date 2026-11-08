---
title: "XLT-11: MAD-X multipole-array import is shadowed by an earlier branch"
hide:
  - toc
---

# XLT-11 — MAD-X multipole-array import is shadowed by an earlier branch

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Source | [first multipole branch, util_programs/mad_to_bmad/madx_to_bmad.py:564–565](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L564-L565)<br>[unreachable array expansion, util_programs/mad_to_bmad/madx_to_bmad.py:629–637](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/mad_to_bmad/madx_to_bmad.py#L629-L637)<br>[bmad/doc/attributes.tex:2045–2058](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/attributes.tex#L2045-L2058) |

Importing a MAD-X multipole with an integrated-strength array produces invalid Bmad attributes. The importer leaves knl and ksl as array-valued fields instead of expanding them into individual multipole orders, so Bmad rejects the generated lattice. For knl = {0, 0, 0.8}, the required coefficient is k2l = 0.8; the output instead contains knl = 0,0,.8. An earlier multipole branch makes the array-expansion branch unreachable.

Run `python XLT-11.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-11.py"

    [Download XLT-11.py](XLT-11.py)

    ```python
    """XLT-11: MAD-X multipole-array import is shadowed by an earlier branch

    Importing a MAD-X multipole with an integrated-strength array produces invalid Bmad attributes. The importer leaves knl and ksl as array-valued fields instead of expanding them into individual multipole orders, so Bmad rejects the generated lattice. For knl = {0, 0, 0.8}, the required coefficient is k2l = 0.8; the output instead contains knl = 0,0,.8. An earlier multipole branch makes the array-expansion branch unreachable.

    The script runs the importer from BMAD_SOURCE on normal and skew arrays, then parses each result with Bmad. Replacing only the array attribute with its individual order provides a parser control; MAD-X also accepts both input arrays.
    """

    import re

    from _bmad import verdict
    from _lattice_import_export import inspect, madx, translate, valid

    defect = False
    for array, field in [("knl", "k2l"), ("ksl", "k2sl")]:
        lattice = (
            f"beam,particle=electron,pc=3; p:multipole,{array}={{0,0,.8}}; ll:line=(p); use,period=ll;"
        )
        with madx() as mad:
            mad.input(lattice)
            assert list(getattr(mad.elements.p, array)) == [0, 0, 0.8]
        status, log, output = translate(lattice)
        if status:
            raise RuntimeError(log)
        result = inspect(output)
        bad = f"BAD ATTRIBUTE NAME: {array.upper()}" in result.log and "PARSE_ERROR  T" in result.log
        if bad:
            corrected, count = re.subn(rf"{array}\s*=\s*0,0,\.8", f"{field}=.8", output)
            assert count == 1
            valid(inspect(corrected))
        else:
            valid(result)
            assert re.search(rf"{field}\s*=\s*\.8", output)
        print(
            f"{array}: parser rejected array={bad}; expected {field}=0.8; individual-order control parsed"
        )
        defect |= bad
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed<br><small class="bb-scope">normal and skew multipole-array import</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

