"""XLT-23: MAD-X sbend k1 export ignores the changed reference momentum

Exporting a combined-function bend after an accelerating cavity leaves its normalized quadrupole gradient k₁ at the local momentum normalization. The MAD-X writer converts the cavity to an RF cavity and renormalizes downstream strengths to the starting momentum, including an ordinary quadrupole with the same k₁. After acceleration from 1 GeV to 2 GeV, k₁ = 0.2 m⁻² exports as 0.4000 m⁻² for the quadrupole but remains 0.2 m⁻² for the bend. Magnetic-rigidity scaling requires 0.4000 m⁻² for both, so the bend’s exported gradient is half the required value.

The script exports matching quadrupole and bend strengths after an lcavity and after a drift control. It derives the required strength ratio from the reference momentum and checks the quadrupole export before comparing the bend.
"""

import math

from _bmad import compare, verdict
from _lattice_import_export import HEADER, definition, inspect, number, records, valid

defect = False
for cavity in [False, True]:
    first = "lcavity,l=1,voltage=1e9,rf_frequency=1e9" if cavity else "drift,l=1"
    lattice = (
        HEADER.replace("parameter[p0c] = 3e9", "parameter[E_tot] = 1e9")
        + f"""c:{first}
q:quadrupole,l=.5,k1=.2
b:sbend,l=1,angle=.01,k1=.2
lat:line=(c,q,b)
use,lat
"""
    )
    result = inspect(lattice, "MAD-X")
    valid(result)
    text = result.outputs["out"].decode()
    p0c = records(result, "P0C")
    ratio = p0c[1][0] / p0c[0][0]
    # Fixed gradient G has k1=qG/p0; normalizing to the initial p0 multiplies by p_local/p_initial.
    mass = 510998.95069
    expected_ratio = math.sqrt(((2 if cavity else 1) * 1e9) ** 2 - mass**2) / math.sqrt(
        1e18 - mass**2
    )
    if compare("reference momentum ratio control", ratio, expected_ratio, atol=0, rtol=3e-14):
        raise RuntimeError("Reference momentum control failed")
    expected = 0.2 * expected_ratio
    if compare(
        f"acceleration={cavity}: quadrupole k1 control",
        number(definition(text, "Q"), "k1"),
        expected,
        atol=0,
        rtol=3e-14,
    ):
        raise RuntimeError("Quadrupole normalization control failed")
    bad = compare(
        f"acceleration={cavity}: sbend k1 (m^-2)",
        number(definition(text, "B"), "k1"),
        expected,
        atol=0,
        rtol=3e-14,
    )
    if not cavity and bad:
        raise RuntimeError("Nonaccelerating bend control failed")
    defect |= bad
verdict(defect)
