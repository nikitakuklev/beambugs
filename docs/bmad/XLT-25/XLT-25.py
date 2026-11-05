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
