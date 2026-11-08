"""DIA-08: `Bunch.ringwrap` moves particles more than one ring length from the reference to the wrong position

`Bunch.ringwrap(L)` maps a longitudinal coordinate z with |z| > L/2 to −sign(z)·L/2 + fmod(z, L/2),
which equals the representative z − L·round(z/L) of z modulo the ring length L only for L/2 < |z| <
L. For |z| between L and 3L/2, between 2L and 5L/2, and so on, the result is shifted by half a ring.
With L = 20 m, z = 25 m wraps to −5 m instead of 5 m and z = 45 m to −5 m instead of 5 m, a 10 m
error. Particles within one ring length of the reference, the case of wrapping on every turn, wrap
correctly.

The script wraps single particles at z = 15, 25, 35, 45, −25 m and a grid of 320 values between
−79.75 m and 79.75 m with L = 20 m and compares each result with z − L·round(z/L), the
representative of z modulo L nearest the reference. Particles with L/2 < |z| < L, the first-lap
case, are the control.
"""

import numpy as np
from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]

L = 20.0


def wrap(zs):
    b = Bunch()
    for z in zs:
        b.addParticle(0.0, 0.0, 0.0, 0.0, float(z), 0.0)
    b.ringwrap(L)
    return np.array([b.z(i) for i in range(b.getSize())])


def expected(z):
    # A ring coordinate is defined modulo L; the wrapped value is the representative in [-L/2, L/2].
    return z - L * np.round(z / L)


ok = True
singles = np.array([15.0, 25.0, 35.0, 45.0, -25.0])
for z, w in zip(singles, wrap(singles)):
    e = expected(z)
    role = " (control: first lap)" if abs(z) < L else ""
    print(f"z = {z:g} m: wrapped observed {w:g} m; expected {e:g} m{role}")
    ok &= (abs(w - e) < 1e-12) == (abs(z) < L or abs(z) == 35.0)

grid = np.arange(-79.75, 80.0, 0.5)
err = np.abs(wrap(grid) - expected(grid))
bad = grid[err > 1e-9]
print(
    f"grid of {grid.size} values: {bad.size} wrapped to the wrong position, all with error "
    f"{sorted({float(v) for v in np.round(err[err > 1e-9], 9)})} m; |z| of the wrong ones lies in "
    f"{sorted({(int(abs(z) // 10) * 10, int(abs(z) // 10) * 10 + 10) for z in bad})} m"
)
ok &= bad.size > 0 and np.allclose(err[err > 1e-9], L / 2)
print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")
