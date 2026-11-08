"""COL-05: `SpaceChargeForceCalc2p5D` gives NaN, wrong kicks or a segmentation fault on grids with nx ≠ ny

`SpaceChargeForceCalc2p5D(nx, ny, nz)` copies its Green function and multiplies the spectra in loops with interchanged x and y bounds, so only square transverse grids index its 2nx × 2ny arrays correctly. Rectangular grids give NaN, finite wrong kicks or a segmentation fault, depending on shape and memory contents. Inside a uniform round beam of radius a the kick is Δx′ = 2Lr₀λx/(β²γ³a²), with L the kick length, r₀ the classical radius and λ the line density. At x = a/2 a 64 × 64 grid gives 1.000 times this kick, a 64 × 63 grid 0.5314 times plus a spurious vertical kick of −0.5649 times, a 64 × 32 grid NaN, and a 32 × 64 grid crashes.

The script tracks a uniform round beam (40000 macroparticles on a sunflower pattern, uniform in z) with two probes at (a/2, 0) and (0, a/2) through `SpaceChargeForceCalc2p5D` on five grid shapes, each in its own child process because some shapes crash. It prints the probe kicks in units of the analytic uniform-beam kick; the square 64 × 64 grid is the control.
"""

import os
import subprocess
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

CHILD = r"""
import sys
import numpy as np
from orbit.core.bunch import Bunch
from orbit.core.spacecharge import SpaceChargeForceCalc2p5D
from orbit.utils.consts import mass_proton
nx, ny = int(sys.argv[1]), int(sys.argv[2])
a, lz, nphys, n, length = 0.01, 1.0, 1e11, 40000, 0.1
k = np.arange(n) + 0.5
r = a * np.sqrt(k / n)                          # sunflower: uniform areal density in r < a
t = k * np.pi * (3.0 - np.sqrt(5.0))
z = (np.arange(n) + 0.5) / n * lz - lz / 2      # uniform in z: lambda = N_phys / lz
b = Bunch(); b.mass(mass_proton); b.macroSize(nphys / n); b.getSyncParticle().kinEnergy(0.1)
for i in range(n):
    b.addParticle(r[i] * np.cos(t[i]), 0, r[i] * np.sin(t[i]), 0, z[i], 0)
ip = b.getSize(); b.addParticle(0.5 * a, 0, 0, 0, 0, 0); b.addParticle(0, 0, 0.5 * a, 0, 0, 0)
SpaceChargeForceCalc2p5D(nx, ny, 8).trackBunch(b, length)
sp = b.getSyncParticle()
lam = nphys * (n + 2) / n / lz
# Uniform round beam: E_r = 2 lambda r / a^2 per r0 inside the beam, so
# dxp = 2 L r0 lambda x / (beta^2 gamma^3 a^2), here at x = a/2.
kx = 2 * length * b.classicalRadius() * lam / (sp.beta() ** 2 * sp.gamma() ** 3 * a**2) * 0.5 * a
print(b.xp(ip) / kx, b.yp(ip) / kx, b.yp(ip + 1) / kx, b.xp(ip + 1) / kx)
"""

bad = False
for nx, ny in ((64, 64), (64, 63), (64, 32), (128, 64), (32, 64)):
    label = f"grid {nx} x {ny}{' (control)' if nx == ny else ''}"
    p = subprocess.run(
        [sys.executable, "-I", "-c", CHILD, str(nx), str(ny)],
        capture_output=True,
        check=False,
        text=True,
        timeout=600,
    )
    if p.returncode != 0:
        print(
            f"{label}: child exit status observed {p.returncode} (crash); expected 0 with kicks 1, 0, 1, 0"
        )
        if nx == ny:
            sys.exit("control grid crashed")
        bad = True
        continue
    xr, yx, yr, xy = (float(v) for v in p.stdout.split())
    print(
        f"{label}: xp(a/2, 0) observed {xr:.4f}, yp(a/2, 0) {yx:.4f}, yp(0, a/2) {yr:.4f}, xp(0, a/2) {xy:.4f} "
        f"(units of the analytic kick); expected 1, 0, 1, 0 (uniform round beam)"
    )
    off = not (abs(xr - 1) < 0.02 and abs(yr - 1) < 0.02 and abs(yx) < 0.02 and abs(xy) < 0.02)
    if nx == ny and off:
        sys.exit("control grid failed")
    bad |= nx != ny and off
print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")
