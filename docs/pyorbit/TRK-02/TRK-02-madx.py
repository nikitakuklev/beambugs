"""TRK-02: MAD-X tracking of the same RFCAVITY, the reference for the kick formula

Setup: needs cpymad (MAD-X), which a PyORBIT environment usually lacks, so it runs in a separate interpreter; it does
not import PyORBIT.

The script tracks one proton at 1 GeV kinetic energy for one turn of a 10 m sequence holding a 1 m RFCAVITY with
VOLT = 0.01 MV and HARMON = 1, at three (LAG, T) pairs, and prints the energy change PT·pc next to
VOLT·10⁻³·sin(2π·LAG − 2πz/C) GeV with z = β·T, the formula the main script uses as its expected value.
"""

import math

from cpymad.madx import Madx

K, M, C = 1.0, 0.938272088, 10.0  # GeV, GeV, m
E = K + M
P = math.sqrt(E * E - M * M)
BETA = P / E

agree = True
for lag, t in ((0.25, 0.0), (0.0, 0.5), (0.1, -0.3)):
    mx = Madx(stdout=False)
    mx.input(f"""
    beam, particle=proton, energy={E};
    rf: rfcavity, l=1.0, volt=0.01, lag={lag}, harmon=1;
    d: drift, l=9.0;
    ring: sequence, l={C}; d, at=4.5; rf, at=9.5; endsequence;
    use, sequence=ring;
    track, onepass, onetable;
    start, x=0, px=0, y=0, py=0, t={t}, pt=0;
    run, turns=1;
    endtrack;
    """)
    de = mx.table.trackone.pt[-1] * P  # PT = dE/(p0 c)
    mx.quit()
    # MAD-X T = -c*dt is positive ahead; PyORBIT z = -beta*c*dt, so z = beta*T
    z = BETA * t
    formula = 0.01e-3 * math.sin(2 * math.pi * lag - 2 * math.pi * z / C)
    print(
        f"[LAG={lag}, T={t} m, z={z:.9f} m] MAD-X dE {de:.9e} GeV, formula VOLT*1e-3*sin(2 pi LAG - 2 pi z/C) {formula:.9e} GeV"
    )
    agree &= abs(de - formula) < 1e-6 * abs(formula)
print("MAD-X AGREES WITH THE FORMULA" if agree else "MAD-X DIFFERS FROM THE FORMULA")
