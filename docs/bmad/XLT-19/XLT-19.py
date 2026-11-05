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
