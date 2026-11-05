"""OPT-23: Radiation-integral and symp_lie_bmad radiation omit the charge squared

For multiply charged ions, radiation_integrals and symp_lie_bmad use a unit-charge radiation
coefficient instead of the species coefficient proportional to Q², where Q is charge in
elementary-charge units. Carbon-12 with Q = 6 loses only 592.2 eV per turn in a test ring
instead of 2.132 × 10⁴ eV, and its linac energy spread is 150.9 eV instead of 905.3 eV.
Damping, emittance growth and stochastic energy variance inherit the missing factor of 36,
while unit-charge controls and the separate native radiation-map loss obey the species scaling.

The script runs the same ring and open line for unit-charge particles and carbon-12. It
compares radiation-integral loss, damping, energy spread and emittance growth with the manual’s
species-radius formulas and checks the native radiation-map loss independently. BMAD_DIST
selects the build; the companion tracking script checks mean loss and fluctuations.
"""

import numpy as np
from _bmad import compare, verdict
from _dispersion_and_moments import RADIATION, execute, radiation_ring, require

defect = False
for species in ["electron", "proton", "#12C+6"]:
    for closed in (True, False):
        rows = execute(RADIATION, {"lat.bmad": radiation_ring(species, closed)})
        mass, energy, gamma, rc, cq, charge = rows["constants"]
        if closed:
            # U0=(2/3)r_c mc^2 gamma^4 I2; damping alpha=U0 J/(2E0).
            require(rows["integrals"][2], 2 * np.pi / (32 / np.pi), atol=0, rtol=2e-14)
            loss = 2 * rc * mass * gamma**4 * rows["integrals"][2] / 3
            expected = np.r_[loss, loss * rows["partition"] / (2 * energy)]
            require(rows["native0"][4], loss, atol=0, rtol=2e-12)
            if abs(charge) == 1:
                require(rows["loss"], expected, atol=0, rtol=2e-14)
            defect |= compare(
                f"{species} ring U0 (eV), alpha_a,b,z from species classical radius",
                rows["loss"],
                expected,
                atol=0,
                rtol=2e-12,
            )
        else:
            # sigma_E^2=(4/3) Cq rc (mc^2)^2 L3; epsilon=(2/3) Cq rc L5/gamma_f.
            spread = np.sqrt(4 * cq * rc * mass**2 * rows["linac"][1] / 3)
            expected = np.r_[spread, 2 * cq * rc * rows["linac"][4:6] / (3 * gamma)]
            observed = rows["linac"][[0, 2, 3]]
            if abs(charge) == 1:
                require(observed, expected, atol=0, rtol=2e-14)
            defect |= compare(
                f"{species} linac sigma_E (eV), epsilon_a,b (m) from species radius",
                observed,
                expected,
                atol=0,
                rtol=2e-12,
            )
verdict(defect)
