"""ELE-07: `FieldTracker` computes the momentum as √(E² − (m·K)²), correct only at K = 1 GeV

`FieldTracker` sets the momentum that scales its field kicks to √(E² − (m·K)²), with E the total energy, m the mass and K
the kinetic energy in GeV, instead of √(E² − m²). The two agree only at K = 1 GeV, so at other energies particles are
bent with the wrong magnetic rigidity. At K = 0.3 GeV a proton-mass particle gets 1.206 GeV/c instead of 0.8081 GeV/c, so
its kicks are 0.6701 of their correct size: in a 0.1 T/m quadrupole gradient over 1 m, a charge −1 particle starting
1 mm off axis leaves with x′ = 2.496×10⁻² mrad instead of 3.733×10⁻² mrad. At K = 1.5 GeV the kicks are 1.130 times too
strong.

The script writes a field map with B = G·(y, x, 0), G = 0.1 T/m, in the tracker's cm and gauss units, builds a
`FieldTracker` from z = 0 to 1 m and tracks a particle of proton mass and charge −1, the tracker's built-in charge, from
x = 1 mm at K = 0.3, 0.6, 1.0 and 1.5 GeV, each case in a child process in a temporary directory because the tracker
writes files into the working directory. It compares the exit x and x′ with the thick-quadrupole map of strength
k = 0.299792458·q·G/p for the true momentum p, and prints the same map with the tracker's momentum beside it. K = 1 GeV
is the control.
"""

import math
import os
import subprocess
import sys
import tempfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

CHILD = r"""
import sys
from orbit.core.bunch import Bunch
from orbit.core.fieldtracker import FieldTracker
q, kin, g_gauss_per_cm = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
with open("field.txt", "w") as f:
    # Header read by the tracker's grid parser: nx ny nz, six skipped records, one skipped integer; then rows
    # x y z [cm] Bx By Bz [gauss]. The linear field B = G (y, x, 0) is exact under trilinear interpolation.
    f.write("3 3 5 0\n" + "1 a b\n" * 6 + "0\n")
    for x in (-20.0, 0.0, 20.0):
        for y in (-12.0, 0.0, 12.0):
            for z in (-10.0, 30.0, 60.0, 90.0, 130.0):
                f.write(f"{x} {y} {z} {g_gauss_per_cm * y} {g_gauss_per_cm * x} 0.0\n")
b = Bunch()
b.mass(0.938272013)
b.charge(q)
b.getSyncParticle().kinEnergy(kin)
b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
b.addParticle(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # the tracker reads x in mm and x' in mrad
# (betaX, betaY, alphaX, alphaY, etaX, etaPX, length, zi, zf, ds, niters, resid, xrefi, yrefi, Euler a, b, g, bunch, file)
ft = FieldTracker(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.001, 3, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, b, "field.txt")
ft.trackBunch(b)
print(f"RESULT {b.x(1)!r} {b.xp(1)!r}")
"""

MASS, Q = 0.938272013, -1.0  # GeV; charge -1, which the tracker uses for every bunch
G, L, X0 = 0.1, 1.0, 1.0  # T/m, m, mm


def thick_quad(k):
    # x'' = -k x from (X0, 0) over L: focusing for k > 0, defocusing for k < 0 (x in mm, x' in mrad)
    if k > 0:
        s = math.sqrt(k)
        return X0 * math.cos(s * L), -X0 * s * math.sin(s * L)
    s = math.sqrt(-k)
    return X0 * math.cosh(s * L), X0 * s * math.sinh(s * L)


def run(kin):
    with tempfile.TemporaryDirectory() as d:
        out = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                CHILD,
                str(Q),
                str(kin),
                str(G * 100.0),
            ],  # 1 T/m = 100 gauss/cm
            cwd=d,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    fields = next(ln for ln in out.splitlines() if ln.startswith("RESULT")).split()
    return float(fields[1]), float(fields[2])


reproduced, control = True, True
for kin in (0.3, 0.6, 1.0, 1.5):
    e = kin + MASS
    p_true = math.sqrt(e * e - MASS * MASS)  # relativistic momentum, GeV/c
    p_code = math.sqrt(
        e * e - (MASS * kin) ** 2
    )  # the tracker's expression; equal to p_true only for kin = 1 GeV
    # By = G x and the Lorentz force F_x = -q v By give x'' = -k x with k = 0.299792458 q G / p [1/m^2].
    ex, exp_ = thick_quad(0.299792458 * Q * G / p_true)
    mx, mxp = thick_quad(0.299792458 * Q * G / p_code)
    x, xp = run(kin)
    dev = max(abs(x - ex), abs(xp - exp_))
    print(
        f"[K = {kin} GeV] p true {p_true:.6f} GeV/c, tracker {p_code:.6f} GeV/c (kick ratio {p_true / p_code:.4f}); "
        f"x observed {x:.6f} mm, expected {ex:.6f} mm; x' observed {xp:+.6e} mrad, expected {exp_:+.6e} mrad; "
        f"max |diff| {dev:.2e}; same map with the tracker's p: x {mx:.6f} mm, x' {mxp:+.6e} mrad"
    )
    if kin == 1.0:
        control &= dev < 1e-8
    else:
        reproduced &= dev > 1e-4 and max(abs(x - mx), abs(xp - mxp)) < 1e-8
print("DEFECT REPRODUCED" if reproduced and control else "NOT REPRODUCED")
