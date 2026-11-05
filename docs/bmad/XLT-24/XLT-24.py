"""XLT-24: MAD export without a reference orbit aborts for a full-fringe quadrupole

Calling the MAD-X lattice writer without its optional reference orbit aborts for a quadrupole with full fringe fields. The writer allocates default coordinates without setting their particle species, then passes one into the fringe-map calculation. The resulting charge lookup encounters the unset species and stops. Supplying the tracked reference orbit succeeds, as does the no-orbit call for the same quadrupole without fringe fields.

The script compiles an inline caller of the foreign-lattice writer and checks tracked-orbit and fringe-free controls. It then calls the full-fringe export without an orbit and recognizes only the unset-species charge failure.
"""

from _bmad import RunFailure, verdict
from _lattice_import_export import HEADER, inspect, valid

lattice = (
    HEADER
    + """q:quadrupole,l=.5,k1=1.2,fringe_type=full
lat:line=(q)
use,lat
"""
)
valid(inspect(lattice, "MAD-X"))
valid(inspect(lattice.replace("fringe_type=full", "fringe_type=none"), "MAD-X", noorbit=True))
print("controls: tracked-orbit full fringe and no-orbit fringe-free exports complete")
try:
    result = inspect(lattice, "MAD-X", noorbit=True)
except RunFailure as exc:
    if (
        "PARSE_ERROR  F" not in exc.log
        or "charge_of:" not in exc.log
        or "CHARGE NOT KNOWN FOR SPECIES: -999" not in exc.log
    ):
        raise
    print(
        f"no-orbit full fringe: status={exc.status}; charge_of rejected unset species; expected completed export"
    )
    print("charge_of: CHARGE NOT KNOWN FOR SPECIES: -999")
    defect = True
else:
    valid(result)
    print("no-orbit full fringe: completed export; expected completed export")
    defect = False
verdict(defect)
