"""IO-05: `dumpBunch` writes mass, charge and `macro_size` with six significant digits

`Bunch.dumpBunch(path)` writes the scalar bunch attributes (mass, charge, `macro_size` and user
doubles) before it sets any output precision, so they carry the stream default of six significant
digits; particle coordinates and attributes are written with eight. A `readBunch` restart therefore
changes the species mass and the macroparticle weight, and the synchronous kinetic energy is
recomputed from the fourteen-digit momentum with the rounded mass. For a 1 GeV proton bunch with mass
0.93827208816 GeV, the reloaded mass is 0.938272 GeV and the reference kinetic energy rises by
45.48 eV (4.548×10⁻⁸ relative); `macro_size` 1.23456789×10¹⁰ reloads as 1.23457×10¹⁰.

The script fills a bunch with a 1 GeV reference, mass 0.93827208816 GeV, `macro_size`
1.23456789×10¹⁰, a user double attribute and one particle, dumps it, reloads it with `readBunch` and
compares each quantity with the value written. The kinetic-energy shift predicted for the rounded
mass is derived in a comment. The control repeats the round trip with a mass of six significant
digits, which the file holds exactly.
"""

import math
import os
import tempfile

from orbit.core.bunch import Bunch

tmp = tempfile.mkdtemp()


def round_trip(mass, label):
    """Dump and reload a 1 GeV bunch; return (reloaded bunch, original kinetic energy, momentum)."""
    b = Bunch()
    b.mass(mass)
    b.macroSize(1.23456789e10)
    b.bunchAttrDouble("user_value", 1.2345678901234567)
    b.getSyncParticle().kinEnergy(1.0)
    b.addParticle(1.234567891e-3, 0.0, 0.0, 0.0, 0.0, 1.23456789012e-4)
    ek0, p0 = b.getSyncParticle().kinEnergy(), b.getSyncParticle().momentum()
    path = os.path.join(tmp, f"{label}.txt")
    b.dumpBunch(path)
    c = Bunch()
    c.readBunch(path)
    return c, ek0, p0


m0 = 0.93827208816  # GeV, CODATA 2018 proton mass
c, ek0, p0 = round_trip(m0, "codata")
m1, ek1 = c.mass(), c.getSyncParticle().kinEnergy()
# A restart file must restore every double it writes; 17 significant digits suffice.
print(f"mass: observed {m1!r} GeV; expected {m0!r} GeV (value written)")
print(f"macro_size: observed {c.macroSize()!r}; expected {1.23456789e10!r} (value written)")
print(
    f"user double: observed {c.bunchAttrDouble('user_value')!r}; expected {1.2345678901234567!r} (value written)"
)
print(f"particle x: observed {c.x(0)!r}; expected {1.234567891e-3!r} (value written)")
# The reader restores p from its 14 digits and recomputes K = sqrt(m^2 + p^2) - m with the rounded
# mass m1, so K shifts by sqrt(m1^2 + p^2) - m1 - K0, to first order (1/gamma - 1)(m1 - m0)
# = (0.48408 - 1)(-8.816e-8 GeV) = +4.548e-8 GeV.
ek_pred = math.sqrt(m1 * m1 + p0 * p0) - m1
print(
    f"reference kinetic energy: observed {ek1!r} GeV (shift {(ek1 - ek0) * 1e9:.2f} eV, "
    f"{ek1 / ek0 - 1:.4g} relative); expected {ek0!r} GeV; rounded-mass prediction {ek_pred!r} GeV"
)

# Control: a mass with six significant digits survives the six-digit field unchanged.
cc, ekc0, _ = round_trip(0.938272, "six_digit")
ekc1 = cc.getSyncParticle().kinEnergy()
print(
    f"six-digit mass (control): observed mass {cc.mass()!r}, K shift {(ekc1 - ekc0) * 1e9:.2f} eV; expected 0.938272, 0.00 eV"
)

defect = m1 != m0 and c.macroSize() != 1.23456789e10 and abs(ek1 - ek_pred) < 1e-13
control = cc.mass() == 0.938272 and abs(ekc1 - ekc0) < 1e-13
print("DEFECT REPRODUCED" if defect and control else "NOT REPRODUCED")
