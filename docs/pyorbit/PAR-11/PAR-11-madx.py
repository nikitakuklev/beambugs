"""PAR-11: MAD-X gives the same solenoid and quadrupole maps for every beam charge

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script loads the 1 m solenoid of the main reproducer (KS = 0.5 m⁻¹) into MAD-X with BEAM particles proton, antiproton, electron and ions of charge ±2 at the same momentum, prints TWISS's transverse transfer matrix for each and tracks a particle from x = 1 mm with TRACK. All matrices and exit points are identical, so MAD-X's KS, like its K1, is normalized for the beam's own charge; a quadrupole with K1 = 0.5 m⁻² is the analogous control, whose charge-independent strength PyORBIT uses as given.
"""

import numpy as np
from cpymad.madx import Madx

BEAMS = [
    "particle=proton",
    "particle=antiproton",
    "particle=electron",
    "particle=ion, mass=0.938272013, charge=2",
    "particle=ion, mass=0.938272013, charge=-2",
]
for elem in ("sol: solenoid, l=1.0, ks=0.5;", "qf: quadrupole, l=1.0, k1=0.5;"):
    name = elem.split(":")[0]
    mats = []
    for beam in BEAMS:
        m = Madx(stdout=False)
        m.input(
            f"{elem}\nseq: sequence, l=1.0, refer=entry; e1: {name}, at=0; endsequence;\n"
            f"beam, {beam}, pc=2.0;\nuse, sequence=seq;"
        )
        tw = m.twiss(betx=1, bety=1, rmatrix=True)
        r = np.array([[tw[f"re{i}{j}"][-1] for j in range(1, 5)] for i in range(1, 5)])
        mats.append(r)
        print(
            f"MAD-X {name} BEAM {beam}: charge {m.sequence.seq.beam.charge:+.0f}, "
            f"R11 {r[0, 0]:+.9f} R13 {r[0, 2]:+.9f} R31 {r[2, 0]:+.9f} R33 {r[2, 2]:+.9f}"
        )
        m.quit()
    spread = max(np.max(np.abs(r - mats[0])) for r in mats)
    print(f"MAD-X {name}: max matrix difference between beam charges {spread:.3e}")

# TRACK through the solenoid from x = 1 mm gives the same exit point for every charge too.
for beam in BEAMS:
    m = Madx(stdout=False)
    m.input(
        "sol: solenoid, l=1.0, ks=0.5;\nseq: sequence, l=1.0, refer=entry; e1: sol, at=0; endsequence;\n"
        f"beam, {beam}, pc=2.0;\nuse, sequence=seq;"
    )
    m.input("track, onepass=true, onetable=true; start, x=1e-3; run, turns=1; endtrack;")
    t = m.table.trackone
    print(
        f"MAD-X TRACK sol BEAM {beam}: exit x {float(t.x[-1]):+.6e} px {float(t.px[-1]):+.6e} "
        f"y {float(t.y[-1]):+.6e} py {float(t.py[-1]):+.6e}"
    )
    m.quit()
