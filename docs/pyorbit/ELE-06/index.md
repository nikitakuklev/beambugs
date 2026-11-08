---
title: "ELE-06: FieldTracker bends every species as a particle of charge −1"
hide:
  - toc
---

# ELE-06 — `FieldTracker` bends every species as a particle of charge −1

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Elements and apertures |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`FieldTracker.cc#L171-L182`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/FieldTracker/FieldTracker.cc#L171-L182)<br>[`FieldTracker.cc#L646-L651`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/FieldTracker/FieldTracker.cc#L646-L651)<br>[`OrbitConst.cc#L34`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/OrbitConst.cc#L34-L34) |

`FieldTracker` converts the field map's magnetic field to a momentum change with the fixed constant `OrbitConst::charge_electron` = −1 instead of the bunch charge number q, so protons and ions follow the trajectory of a charge −1 particle of the same momentum. In a field with a 0.1 T/m quadrupole gradient over 1 m, a proton of 1 GeV kinetic energy starting 1 mm off axis is defocused to x = 1.009 mm where it should be focused to 0.9912 mm. A q = +2 ion also gives 1.009 mm instead of 0.9824 mm. Reversing the field reverses the kick, so only the charge is ignored.

Run `python ELE-06.py`.

??? example "Reproducer: ELE-06.py"

    [Download ELE-06.py](ELE-06.py)

    ```python
    """ELE-06: `FieldTracker` bends every species as a particle of charge −1

    `FieldTracker` converts the field map's magnetic field to a momentum change with the fixed constant
    `OrbitConst::charge_electron` = −1 instead of the bunch charge number q, so protons and ions follow the trajectory of a
    charge −1 particle of the same momentum. In a field with a 0.1 T/m quadrupole gradient over 1 m, a proton of 1 GeV
    kinetic energy starting 1 mm off axis is defocused to x = 1.009 mm where it should be focused to 0.9912 mm. A q = +2
    ion also gives 1.009 mm instead of 0.9824 mm. Reversing the field reverses the kick, so only the charge is ignored.

    The script writes a field map with B = G·(y, x, 0), G = 0.1 T/m, in the tracker's cm and gauss units, builds a
    `FieldTracker` from z = 0 to 1 m and tracks a particle of 1 GeV kinetic energy starting at x = 1 mm for q = +1, +2 and
    −1, and for q = ±1 in the reversed field, each case in a child process in a temporary directory because the tracker
    writes files into the working directory. It compares the exit x and x′ with the thick-quadrupole map of strength
    k = 0.299792458·q·G/p. The kinetic energy is 1 GeV, the only one at which the tracker's momentum is right; the two
    q = −1 rows are the controls.
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

    MASS, KIN = 0.938272013, 1.0  # GeV
    G, L, X0 = 0.1, 1.0, 1.0  # T/m, m, mm
    P = math.sqrt((KIN + MASS) ** 2 - MASS**2)  # GeV/c


    def thick_quad(k):
        # x'' = -k x from (X0, 0) over L: focusing for k > 0, defocusing for k < 0 (x in mm, x' in mrad)
        if k > 0:
            s = math.sqrt(k)
            return X0 * math.cos(s * L), -X0 * s * math.sin(s * L)
        s = math.sqrt(-k)
        return X0 * math.cosh(s * L), X0 * s * math.sinh(s * L)


    def run(q, g):
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    CHILD,
                    str(q),
                    str(KIN),
                    str(g * 100.0),
                ],  # 1 T/m = 100 gauss/cm
                cwd=d,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        fields = next(ln for ln in out.splitlines() if ln.startswith("RESULT")).split()
        return float(fields[1]), float(fields[2])


    reproduced, controls = True, True
    for q, g, label in (
        (1.0, G, "proton"),
        (1.0, -G, "proton, reversed field"),
        (2.0, G, "q = +2"),
        (-1.0, G, "q = -1, control"),
        (-1.0, -G, "q = -1, reversed field, control"),
    ):
        # By = G x and the Lorentz force F_x = -q v By give x'' = -k x with k = 0.299792458 q G / p [1/m^2].
        k = 0.299792458 * q * g / P
        ex, exp_ = thick_quad(k)
        x, xp = run(q, g)
        dev = max(abs(x - ex), abs(xp - exp_))
        print(
            f"[{label}, G = {g:+.1f} T/m] x observed {x:.6f} mm, expected {ex:.6f} mm; x' observed {xp:+.6e} mrad, "
            f"expected {exp_:+.6e} mrad (thick quadrupole, k = {k:+.5e} m^-2); max |diff| {dev:.2e}"
        )
        if "control" in label:
            controls &= dev < 1e-8
        else:
            reproduced &= dev > 1e-3
    print("DEFECT REPRODUCED" if reproduced and controls else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

