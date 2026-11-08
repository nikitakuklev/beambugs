"""PAR-12: MAD-X tracks the tilted elements of the main reproducer as rolled by +TILT

Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

The script places the tilted kicker, quadrupole and thin multipole of the main reproducer in the same 3 m line and computes the exit orbit from x = 1 mm, y = 2 mm with TWISS, which matches the main reproducer's expected values for an element rolled by +TILT. It also tracks a particle with dE = 1 MeV through the 1 m sector bend with TILT = π/2, which leaves at positive y.
"""

import math

from cpymad.madx import Madx

E = 1.0 + 0.93827208816  # total energy of a 1 GeV proton
CASES = (
    ("kicker HKICK=1e-3 TILT=0.4", "kicker, l=0, hkick=1e-3, tilt=0.4"),
    ("quadrupole K1=0.5 TILT=0.3", "quadrupole, l=1.0, k1=0.5, tilt=0.3"),
    ("multipole KNL={0, 0.3} TILT=0.7", "multipole, knl={0, 0.3}, tilt=0.7"),
)
for label, element in CASES:
    m = Madx(stdout=False)
    m.input(
        f"beam, particle=proton, energy={E!r};\ne1: {element};\nseq: sequence, l=3, refer=centre;\ne1, at=1.5;\nendsequence;\nuse, sequence=seq;"
    )
    tw = m.twiss(betx=10, bety=10, x=1e-3, px=0, y=2e-3, py=0)
    exit_ = ", ".join(f"{float(tw[k][-1]):+.6e}" for k in ("x", "px", "y", "py"))
    print(f"MAD-X [{label}] exit (x, px, y, py) ({exit_})")
    m.quit()

m = Madx(stdout=False)
m.input(
    f"beam, particle=proton, energy={E!r};\ne1: sbend, l=1.0, angle=0.1, tilt={math.pi / 2!r};\n"
    "seq: sequence, l=3, refer=centre;\ne1, at=1.5;\nendsequence;\nuse, sequence=seq;"
)
p0 = math.sqrt(E * E - 0.93827208816**2)
m.input(f"track, onepass=true, onetable=true; start, pt={1e-3 / p0!r}; run, turns=1; endtrack;")
t = m.table.trackone
print(
    f"MAD-X [sbend ANGLE=0.1 TILT=pi/2, dE = 1 MeV] exit x {float(t.x[-1]):+.6e} m, y {float(t.y[-1]):+.6e} m"
)
m.quit()
