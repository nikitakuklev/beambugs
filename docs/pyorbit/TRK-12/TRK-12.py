"""TRK-12: The barrier RF cavity gives every species the energy kick of a unit positive charge

`Barrier_RFNode` and `TimeDep_Barrier_RFNode` add their barrier waveform ±V·cos(π(φ − φ±)/(2Δφ±)) to each particle's dE without the bunch charge number q, with V the voltage in GV, φ the particle's RF phase and φ±, Δφ± the barrier centres and half-widths; the harmonic RF nodes, whose voltage is documented in the same words, add q·V. A particle of charge q crossing a voltage V gains q·V, so a q = −1 particle receives a proton's kick with the wrong sign, and a q = +28 ion 1/28 of its kick. For a 10⁻⁵ GV barrier and a particle 10° from the centre of the positive barrier, q = −1 gains +8.660×10⁻⁶ GeV instead of −8.660×10⁻⁶ GeV, and q = +28 gains 8.660×10⁻⁶ GeV instead of 2.425×10⁻⁴ GeV.

The script places particles at RF phases +160°, −160° and +100° (inside the positive barrier, inside the negative one, outside both) and tracks them once through `Barrier_RFNode` and through `TimeDep_Barrier_RFNode` with V = 10⁻⁵ GV and barriers centred at ±150° with half-widths of 30°, for bunches of charge +1, −1, +2 and +28. It compares dE with q·V·cos(π(φ − φ±)/(2Δφ±)). The q = +1 rows and `Harmonic_RFNode` of the same voltage, whose kick scales with q, are the controls.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core.bunch import Bunch
from orbit.rf_cavities import RFNode

ZTOPHI = 2.0 * math.pi / 4.0  # 4 m ring, as in the upstream barrier test
V = 1e-5  # GV
PHP, PHM, DPH = 150.0, -150.0, 30.0  # barrier centres and half-widths, deg
PHASES = (160.0, -160.0, 100.0)  # particle RF phases, deg
SPECIES = (
    ("proton", 1.0, 0.938272088),
    ("antiproton", -1.0, 0.938272088),
    ("q=+2", 2.0, 0.938272088),
    ("U28+", 28.0, 238 * 0.9314941),
)
TIMEDEP = {
    "n_tuple": 1,
    "time": (0.0, 1.0),
    "RFVoltage": (V, V),
    "RFPhasep": (PHP, PHP),
    "RFPhasem": (PHM, PHM),
    "dRFPhasep": (DPH, DPH),
    "dRFPhasem": (DPH, DPH),
}


def bunch(q, mass):
    b = Bunch()
    b.mass(mass)
    b.charge(q)
    b.getSyncParticle().kinEnergy(1.0)
    b.getSyncParticle().time(0.0)
    for ph in PHASES:
        # the cavities use the RF phase phi = -ZtoPhi*z, so a particle at phase phi sits at z = -phi/ZtoPhi
        b.addParticle(0.0, 0.0, 0.0, 0.0, -math.radians(ph) / ZTOPHI, 0.0)
    return b


def expected(q, ph):
    # A charge q crossing the barrier voltage gains q*(+V cos(arg)) in the positive barrier and q*(-V cos(arg))
    # in the negative one, arg = pi (phi - phi_centre)/(2 dphi); nothing outside both windows.
    if abs(ph - PHP) < DPH:
        return q * V * math.cos(math.pi * (ph - PHP) / (2 * DPH))
    if abs(ph - PHM) < DPH:
        return -q * V * math.cos(math.pi * (ph - PHM) / (2 * DPH))
    return 0.0


mismatches, controls_ok = 0, True
for name, q, mass in SPECIES:
    for path in ("Barrier_RFNode", "TimeDep_Barrier_RFNode"):
        b = bunch(q, mass)
        if path == "Barrier_RFNode":
            RFNode.Barrier_RFNode(ZTOPHI, V, PHP, PHM, DPH, DPH, 0.0, "barrier").trackBunch(b)
        else:
            RFNode.TimeDep_Barrier_RFNode(ZTOPHI, TIMEDEP, b, 0.0, "barrier").trackBunch(b)
        for i, ph in enumerate(PHASES):
            exp = expected(q, ph)
            bad = abs(b.dE(i) - exp) > 1e-12 * V
            print(
                f"[{name:10s} q={q:+5.1f} {path:22s}] phi {ph:+6.1f} deg: dE observed {b.dE(i):+.6e} GeV, "
                f"expected {exp:+.6e} GeV (q V cos(pi (phi - phi_c)/(2 dphi)))"
                f"{' (control)' if q == 1.0 else ''}"
            )
            if q == 1.0:
                controls_ok &= not bad
            else:
                mismatches += bad
    # Control: Harmonic_RFNode (h = 1, phase 0, no synchronous gain) adds q V sin(phi) at phi = +160 deg.
    b = bunch(q, mass)
    RFNode.Harmonic_RFNode(ZTOPHI, 0.0, 1.0, V, 0.0, 0.0, "harmonic").trackBunch(b)
    exp = q * V * math.sin(math.radians(PHASES[0]))
    print(
        f"[{name:10s} q={q:+5.1f} Harmonic_RFNode       ] phi {PHASES[0]:+6.1f} deg: dE observed {b.dE(0):+.6e} GeV, "
        f"expected {exp:+.6e} GeV (q V sin(phi), control)"
    )
    controls_ok &= abs(b.dE(0) - exp) < 1e-12 * V

# 3 charges x 2 paths x 2 phases inside a barrier
print("DEFECT REPRODUCED" if mismatches == 12 and controls_ok else "NOT REPRODUCED")
