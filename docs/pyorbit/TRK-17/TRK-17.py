"""TRK-17: `BRhoDep_Harmonic_RFNode` stores the unstable synchronous position when the charge times the voltage is negative

`BRhoDep_Harmonic_RFNode` stores as the synchronous z the position of the phase φs = arcsin(ΔEs/(qV)) below transition and π − arcsin above, with ΔEs the reference energy gain per pass, q the charge number and V the voltage, whatever the sign of qV. The cavity kick is qV sin φ, so for qV < 0 the stored z is the unstable fixed point. In a 248 m ring at 1 GeV with harmonic 4, 10 kV and q = −1, the stored z is +0.7329 m and the stable point −31.73 m; over 3000 turns a particle started 5 cm from the stored z runs 52.18 m away, one 5 cm from the stable point stays within 5.000 cm. The cavity kick and reference energy update are correct for either sign.

The script builds the node for a proton and for a q = −1 particle of proton mass with the same voltage and rigidity ramp and reads the stored synchronous z. It computes both fixed points of the kick, picks the stable one from the sign of the kick's slope, and tracks a particle started 5 cm from each through a ring of one drift and `Harmonic_RFNode` with the node's synchronous gain for 3000 turns (about three synchrotron periods) at fixed reference energy. The proton, whose stored z is the stable point, is the control.
"""

import math
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from orbit.core.bunch import Bunch
from orbit.rf_cavities import RFNode
from orbit.teapot import DriftTEAPOT

C, H, V, RFPH, GT = (
    248.0,
    4.0,
    1e-5,
    0.0,
    5.0,
)  # ring length (m), harmonic, voltage (GV), RF phase (deg), gamma_t
ZTOPHI = 2.0 * math.pi / C
TURNS, DZ0 = 3000, 0.05


def excursion(z_start, q, des):
    # max |z - z_start| of a particle started DZ0 ahead of z_start; one turn is the ring drift and the cavity
    t = Bunch()
    t.mass(0.938272013)
    t.charge(q)
    t.getSyncParticle().kinEnergy(1.0)
    t.addParticle(0.0, 0.0, 0.0, 0.0, z_start + DZ0, 0.0)
    drift = DriftTEAPOT("ring")
    drift.setLength(C)
    cav = RFNode.Harmonic_RFNode(ZTOPHI, des, H, V, RFPH, 0.0, "cav")
    worst = 0.0
    for _ in range(TURNS):
        drift.trackBunch(t)
        cav.trackBunch(t)
        worst = max(worst, abs(t.z(0) - z_start))
    return worst


defects, controls_ok = 0, True
for q in (1.0, -1.0):
    b = Bunch()
    b.mass(0.938272013)
    b.charge(q)
    b.getSyncParticle().kinEnergy(1.0)
    b.getSyncParticle().time(0.0)
    brho0 = 3.335640952 * b.getSyncParticle().momentum()  # |B rho| of the 1 GeV reference, T m
    acc = dict(
        gammaTrans=GT,
        RFHNum=H,
        n_tuple=1,
        time=(0.0, 1.0),
        BRho=(brho0 * 1.0000005, brho0 * 1.0001),
        RFVoltage=(V, V),
        RFPhase=(RFPH, RFPH),
    )
    node = RFNode.BRhoDep_Harmonic_RFNode(ZTOPHI, acc, b, 0.0, "rf")
    z_stored = b.getSyncParticle().z()
    des = node.harmonicnode.dESync()
    gamma = b.getSyncParticle().gamma()
    # Fixed points of the kick q V sin(phi) = dEs with phi = -h ZtoPhi z + phi_RF: phi = asin(x) and pi - asin(x),
    # x = dEs/(qV). Below transition (gamma < gamma_t) a particle ahead (larger z) arrives earlier, so the stable
    # point needs d(kick)/dz = -q V h ZtoPhi cos(phi) < 0.
    x = des / (q * V)
    roots = [
        -(ph - math.radians(RFPH)) / (H * ZTOPHI) for ph in (math.asin(x), math.pi - math.asin(x))
    ]
    slopes = [-q * V * H * ZTOPHI * math.cos(-H * ZTOPHI * z + math.radians(RFPH)) for z in roots]
    z_stable = roots[0] if slopes[0] < 0 else roots[1]
    exc_stored, exc_stable = excursion(z_stored, q, des), excursion(z_stable, q, des)
    label = "proton (control)" if q > 0 else "q = -1"
    print(
        f"[{label}] gamma {gamma:.4f} < gamma_t {GT}, dEs {des:.6e} GeV: stored synchronous z observed "
        f"{z_stored:+.6f} m, expected {z_stable:+.6f} m (stable fixed point); max |z - z0| over {TURNS} turns from "
        f"{DZ0} m: from the stored z {exc_stored:.4f} m, from the stable point {exc_stable:.4f} m"
    )
    wrong = abs(z_stored - z_stable) > 1e-9 and exc_stored > 10 * DZ0
    if q > 0:
        controls_ok &= not wrong and exc_stored < 1.01 * DZ0
    else:
        defects += wrong
        controls_ok &= exc_stable < 1.01 * DZ0
print("DEFECT REPRODUCED" if defects == 1 and controls_ok else "NOT REPRODUCED")
