"""OPT-22: Coupling integral reads a KSEXT's YAW as its vertical offset DY

Elegant's `twiss_output` coupling estimate (`couplingIntegral`, `emittanceRatio`) includes the skew
gradient K2·(y − DY) of vertically displaced sextupoles, but for a KSEXT it reads DY through the
SEXT data layout, where the KSEXT stores its YAW. A vertically displaced KSEXT therefore contributes
no coupling, and a yawed one contributes as if displaced by its YAW in meters. In an eight-cell FODO
ring with one 0.08 m sextupole of K2 = 4 m⁻³ displaced by DY = 0.7 mm, the single-source formula and
a SEXT both give a coupling integral of 3.983×10⁻⁴; the equivalent KSEXT gives 0, and a KSEXT with
YAW = 0.7 mrad and no displacement gives 3.983×10⁻⁴ instead of 0.

The script runs serial Elegant `twiss_output` on an eight-cell FODO ring with one 0.08 m sextupole of
K2 = 4 m⁻³ and reads `couplingIntegral` and `emittanceRatio` for a SEXT with DY = 0.7 mm, a KSEXT
with the same DY and a KSEXT with YAW = 0.7 mrad. The expected coupling integral of a single skew
source is |K2·DY|·L·√(β̄x β̄y)/(2π), with β̄ the average of the beta functions at the sextupole's
ends (the convention of Elegant's sum). Second-order matrices (`matrix_output`) show that the SEXT
and the displaced KSEXT have the same x–y coupling term R13 and that the yawed KSEXT has none.
"""

import sys

import numpy as np
from _elegant import run, verdict

K2, LS, OFFSET = 4.0, 0.08, 7e-4


def ring(element, misalignment):
    lattice = (
        "F: QUAD,L=0.25,K1=1.1\nG: QUAD,L=0.25,K1=-1\nD: DRIF,L=0.7\nE: DRIF,L=0.9\n"
        f"S: {element},L={LS},K2={K2},{misalignment}\nC: LINE=(F,D,G,E)\nBL: LINE=(4*C,S,4*C)\n"
    )
    commands = (
        "&run_setup lattice=lattice.lte, use_beamline=BL, p_central_mev=2400, default_order=2 &end\n"
        "&twiss_output filename=twiss.sdds, matched=1 &end\n"
        "&matrix_output SDDS_output=matrix.sdds, SDDS_output_order=2, full_matrix_only=1 &end\n"
    )
    result = run(
        {"lattice.lte": lattice, "run.ele": commands}, outputs=("twiss.sdds", "matrix.sdds")
    )
    twiss = result["twiss.sdds"]
    names = list(twiss[0, "ElementName"])
    i = names.index("S")
    # Average of the beta functions at the sextupole's entrance (row i-1) and exit (row i).
    beta = [float(np.mean(twiss[0, f"beta{p}"][i - 1 : i + 1])) for p in "xy"]
    values = [float(twiss[k][0]) for k in ("couplingIntegral", "emittanceRatio", "couplingDelta")]
    r13 = float(result["matrix.sdds"][0, "R13"][-1])
    return beta, values, r13


defect = False
cases = [
    ("SEXT DY=0.7 mm", "SEXT", f"DY={OFFSET!r}", True),
    ("KSEXT DY=0.7 mm", "KSEXT", f"DY={OFFSET!r}", True),
    ("KSEXT YAW=0.7 mrad", "KSEXT", f"YAW={OFFSET!r}", False),
]
r13 = {}
for label, element, misalignment, displaced in cases:
    beta, (integral, ratio, delta), r13[label] = ring(element, misalignment)
    # Skew gradient of a vertically displaced sextupole: K1r = K2 (y - DY) with y = 0 on axis.
    kappa = abs(K2 * OFFSET) * LS * np.sqrt(beta[0] * beta[1]) / (2 * np.pi) if displaced else 0.0
    # emittanceRatio = kappa^2 / (kappa^2 + couplingDelta^2), couplingDelta the tune distance.
    expected_ratio = kappa**2 / (kappa**2 + delta**2)
    print(
        f"{label} couplingIntegral, emittanceRatio: observed=[{integral:.10g}, {ratio:.10g}]"
        f" expected=[{kappa:.10g}, {expected_ratio:.10g}]; R13={r13[label]:.10g} m"
    )
    bad = not np.isclose(integral, kappa, rtol=1e-9, atol=1e-15)
    if element == "SEXT" and bad:
        sys.exit("The SEXT control disagrees with the single-source formula: invalid comparison.")
    defect |= bad
if (
    not np.isclose(r13["SEXT DY=0.7 mm"], r13["KSEXT DY=0.7 mm"], rtol=1e-9)
    or r13["KSEXT YAW=0.7 mrad"] != 0
):
    sys.exit("The R13 control does not show the expected element equivalence: invalid comparison.")
print("control: SEXT and displaced KSEXT have the same R13; the yawed KSEXT has R13 = 0")
verdict(defect)
