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
