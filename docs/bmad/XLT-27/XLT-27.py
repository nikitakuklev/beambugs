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
