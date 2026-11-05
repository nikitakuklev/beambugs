"""XLT-18: noaperture documentation contradicts the MAD-X output guard

The translator documentation says the noaperture option is ignored for MAD-X non-drift elements that can carry aperture attributes. In fact, using the option suppresses those attributes. A quadrupole with 0.01 m and 0.02 m half-apertures exports both limits normally and neither with noaperture. The option documentation and the output guard therefore specify different behavior.

The script reads the CLI and writer contracts from BMAD_SOURCE and exports the same quadrupole with and without noaperture. It checks the default aperture values and reports whether the documented exception agrees with the option output.
"""

import re

from _bmad import compare, source, verdict
from _lattice_import_export import HEADER, definition, export

root = source()
driver = (root / "util_programs/bmad_to_mad_sad_elegant/bmad_to_mad_sad_elegant.f90").read_text()
writer = (root / "bmad/output/write_lattice_mad_format.f90").read_text()
driver_contract = "in this case this argument is ignored." in driver
writer_contract = "with an aperture so in this case this argument is ignored." in writer
if not driver_contract or not writer_contract:
    raise RuntimeError("The aperture contract changed; inspect the new documentation")
lattice = (
    HEADER
    + """q:quadrupole,l=.5,k1=.2,x_limit=.01,y_limit=.02
lat:line=(q)
use,lat
"""
)
normal = definition(export(lattice), "Q")
option = definition(export(lattice, noaperture=True), "Q")
match = re.search(r"aperture\s*=\s*\{([^}]+)\}", normal, re.IGNORECASE)
if match is None:
    raise RuntimeError(normal)
if compare(
    "default half-apertures (m)", [float(v) for v in match[1].split(",")], [0.01, 0.02], atol=0
):
    raise RuntimeError("Default apertures incorrect")
missing = re.search(r"\baperture\s*=", option, re.IGNORECASE) is None
print(f"noaperture: aperture attributes present={not missing}; documented expected=True")
verdict(missing)
