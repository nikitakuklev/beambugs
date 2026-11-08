---
title: "XLT-19: Five Elegant importer mappings emit invalid Bmad element classes"
hide:
  - toc
---

# XLT-19 — Five Elegant importer mappings emit invalid Bmad element classes

| | |
|---|---|
| Code | [Bmad](../index.md) · Translators |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad release [20261031-0](https://github.com/bmad-sim/bmad-ecosystem/tree/20261031-0) (`c4c629f`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [ele_type_translate mappings, util_programs/elegant_to_bmad/elegant_to_bmad.py:79–169](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/elegant_to_bmad/elegant_to_bmad.py#L79-L169)<br>[parse_element lookup and serialization, util_programs/elegant_to_bmad/elegant_to_bmad.py:567–600](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/elegant_to_bmad/elegant_to_bmad.py#L567-L600)<br>[Bmad key lookup and rejection, bmad/parsing/bmad_parser.f90:733–752](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/parsing/bmad_parser.f90#L733-L752)<br>[announces beta status](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/util_programs/elegant_to_bmad/elegant_to_bmad.py#L993) |

Importing Elegant SEXT, OCTU, SLICE, EVKICK, or RFTMEZ0 elements emits misspelled Bmad element classes. The resulting classes are sextuple, octuple, maker, vhicker, and rf_cavity, respectively. Bmad rejects every generated definition as an unrecognized key. Changing only the class to sextupole, octupole, marker, vkicker, or rfcavity makes each lattice parse.

Run `python XLT-19.py` with [_bmad.py](../helpers/_bmad.py) and [_lattice_import_export.py](../helpers/_lattice_import_export.py) in the same directory.

??? example "Reproducer: XLT-19.py"

    [Download XLT-19.py](XLT-19.py)

    ```python
    """XLT-19: Five Elegant importer mappings emit invalid Bmad element classes

    Importing Elegant SEXT, OCTU, SLICE, EVKICK, or RFTMEZ0 elements emits misspelled Bmad element classes. The resulting classes are sextuple, octuple, maker, vhicker, and rf_cavity, respectively. Bmad rejects every generated definition as an unrecognized key. Changing only the class to sextupole, octupole, marker, vkicker, or rfcavity makes each lattice parse.

    The script invokes the Elegant importer from BMAD_SOURCE for each affected class, then calls the Bmad parser. Each control changes only the generated class spelling and requires a successful parse.
    """

    from _bmad import verdict
    from _lattice_import_export import HEADER, inspect, translate, valid

    defect = False
    for incoming, bad, good in [
        ("SEXT", "sextuple", "sextupole"),
        ("OCTU", "octuple", "octupole"),
        ("SLICE", "maker", "marker"),
        ("EVKICK", "vhicker", "vkicker"),
        ("RFTMEZ0", "rf_cavity", "rfcavity"),
    ]:
        status, log, output = translate(f"e:{incoming}\nlat:LINE=(e)\n", "elegant")
        if status:
            raise RuntimeError(log)
        result = inspect(HEADER + output)
        failed = (
            f"KEY NAME NOT RECOGNIZED OR AMBIGUOUS: {bad.upper()}" in result.log
            and "PARSE_ERROR  T" in result.log
        )
        if failed:
            assert f"e: {bad}," in output
            control = inspect(HEADER + output.replace(f"e: {bad},", f"e: {good},"))
            valid(control)
            assert f"ELEMENT 1 E {good.upper()}" in control.log.upper()
        else:
            valid(result)
            assert f"ELEMENT 1 E {good.upper()}" in result.log.upper()
        print(
            f"{incoming}: invalid class {bad} rejected={failed}; expected {good}; corrected-class control parsed"
        )
        defect |= failed
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-20</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-21</span> | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed<br><small class="bb-scope">five Elegant-to-Bmad element-class mappings</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-05</span> | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

