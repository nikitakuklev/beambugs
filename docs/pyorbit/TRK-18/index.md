---
title: "TRK-18: A weak TEAPOT solenoid loses the transverse positions through cancellation"
hide:
  - toc
---

# TRK-18 — A weak TEAPOT solenoid loses the transverse positions through cancellation

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">extreme-regime</span> [?](../../methodology.md#tags) |
| Source | [`teapotbase.cc#L1367-L1413`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L1367-L1413)<br>[`teapot.py#L701-L716`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L701-L716)<br>[`LinacAccNodes.py#L881-L893`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L881-L893) |

The TEAPOT solenoid kernel, used by `SolenoidTEAPOT` and the linac `Solenoid`, switches to a drift only for a strength |B| < 10⁻¹⁰⁰ m⁻¹, B being the field over the rigidity. For weaker strengths it recovers the positions by subtracting terms of size p⊥/B, with p⊥ the transverse momentum, so cancellation leaves a position error of about 10⁻¹⁶·p⊥/B. For a 1 GeV proton with x = 1 mm, pₓ = 1 mrad, y = 0.5 mm and p_y = 2 mrad through 1 m, x is off by 1.362×10⁻⁷ m at B = 10⁻¹² m⁻¹ and 3.830×10⁻⁵ m at 10⁻¹⁴ m⁻¹, and for |B| ≤ 10⁻¹⁷ m⁻¹ the particle leaves at x = y = 0 instead of 2.000 mm and 2.500 mm. At 10⁻⁸ m⁻¹ (56.57 nT) the error is 2.049×10⁻¹¹ m.

Run `python TRK-18.py`.

??? example "Reproducer: TRK-18.py"

    [Download TRK-18.py](TRK-18.py)

    ```python
    """TRK-18: A weak TEAPOT solenoid loses the transverse positions through cancellation

    The TEAPOT solenoid kernel, used by `SolenoidTEAPOT` and the linac `Solenoid`, switches to a drift only for a strength |B| < 10⁻¹⁰⁰ m⁻¹, B being the field over the rigidity. For weaker strengths it recovers the positions by subtracting terms of size p⊥/B, with p⊥ the transverse momentum, so cancellation leaves a position error of about 10⁻¹⁶·p⊥/B. For a 1 GeV proton with x = 1 mm, pₓ = 1 mrad, y = 0.5 mm and p_y = 2 mrad through 1 m, x is off by 1.362×10⁻⁷ m at B = 10⁻¹² m⁻¹ and 3.830×10⁻⁵ m at 10⁻¹⁴ m⁻¹, and for |B| ≤ 10⁻¹⁷ m⁻¹ the particle leaves at x = y = 0 instead of 2.000 mm and 2.500 mm. At 10⁻⁸ m⁻¹ (56.57 nT) the error is 2.049×10⁻¹¹ m.

    The script tracks one particle through `teapot_base.soln` at strengths from 10⁻² to 10⁻⁵⁰ m⁻¹ and compares x and y with the solenoid matrix written without cancellation, sin(BL)/B and 2 sin²(BL/2)/B, cross-checked against the kernel's own formula evaluated in 60-digit decimal arithmetic. It repeats the 10⁻¹⁷ m⁻¹ case through `SolenoidTEAPOT` in a `TEAPOT_Lattice` and the linac `Solenoid` in a `LinacAccLattice`. The B = 0 drift and the B = 10⁻² m⁻¹ solenoid are the controls.
    """

    import math
    import os
    from decimal import Decimal as D
    from decimal import getcontext

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import orbit.core.teapot_base as TPB
    from orbit.core.bunch import Bunch
    from orbit.py_linac.lattice import LinacAccLattice, Sequence, Solenoid
    from orbit.teapot import SolenoidTEAPOT, TEAPOT_Lattice

    getcontext().prec = 60
    MASS, EKIN, L = 0.938272013, 1.0, 1.0  # GeV, GeV, m
    X0 = (1e-3, 1e-3, 0.5e-3, 2e-3, 0.0, 0.0)  # x, px, y, py, z, dE


    def bunch():
        b = Bunch()
        b.mass(MASS)
        b.getSyncParticle().kinEnergy(EKIN)
        b.addParticle(*X0)
        return b


    def kernel(b_field):
        b = bunch()
        TPB.soln(b, L, b_field, 1)
        return b.x(0), b.y(0)


    def teapot_node(b_field):
        lat = TEAPOT_Lattice("line")
        lat.addNode(SolenoidTEAPOT("s", B=b_field, length=L))
        lat.initialize()
        b = bunch()
        lat.trackBunch(b)
        return b.x(0), b.y(0)


    def linac_node(b_field):
        lat = LinacAccLattice("line")
        seq = Sequence("seq")
        seq.setLinacAccLattice(lat)
        node = Solenoid("s")
        node.setLength(L)
        node.setParam("B", b_field)
        seq.addNode(node)
        lat.addNode(node)
        lat.initialize()
        b = bunch()
        lat.trackBunch(b)
        return b.x(0), b.y(0)


    def matrix(b_field):
        # Solenoid map in the kernel's canonical variables (Larmor angle BL/2, dE = 0), written without cancellation:
        # (1 - cos BL)/B = 2 sin^2(BL/2)/B and sin(BL)/B = L sinc(BL); it tends to the drift as B -> 0.
        x, px, y, py = X0[:4]
        ph = b_field * L
        c = math.cos(ph)
        s_over_b = L * (math.sin(ph) / ph if ph != 0.0 else 1.0)
        omc_over_b = 2.0 * math.sin(ph / 2) ** 2 / b_field if b_field != 0.0 else 0.0
        xf = (1 + c) / 2 * x + s_over_b * px + math.sin(ph) / 2 * y + omc_over_b * py
        yf = -math.sin(ph) / 2 * x - omc_over_b * px + (1 + c) / 2 * y + s_over_b * py
        return xf, yf


    def dtrig(x):
        # sin and cos by Taylor series in 60-digit decimal arithmetic
        s, c, t, n = D(0), D(0), D(1), 0
        while abs(t) > D(10) ** -70 or n < 2:
            if n % 4 == 0:
                c += t
            elif n % 4 == 1:
                s += t
            elif n % 4 == 2:
                c -= t
            else:
                s -= t
            n += 1
            t = t * x / n
        return s, c


    def kernel_in_60_digits(b_field):
        # teapot_base::soln with charge +1 and dE = 0 (KNL = 1), every operation in 60 digits
        x, px, y, py = (D(v) for v in X0[:4])
        bc = D(b_field)
        cu, cpu = y / 2 - px / bc, x * bc / 2 + py
        u0, pu0 = y / 2 + px / bc, -x * bc / 2 + py
        sn, cs = dtrig(bc * D(L))
        u = u0 * cs + pu0 * sn / bc
        pu = -u0 * bc * sn + pu0 * cs
        return float((-pu + cpu) / bc), float(u + cu)


    defects, controls_ok = 0, True
    xd, yd = X0[0] + L * X0[1], X0[2] + L * X0[3]
    x, y = kernel(0.0)
    print(
        f"[soln B = 0 (control)] x observed {x:.12e} m, expected {xd:.12e}; y observed {y:.12e}, expected {yd:.12e} (drift)"
    )
    controls_ok &= abs(x - xd) < 1e-18 and abs(y - yd) < 1e-18
    for b_field in (1e-2, 1e-8, 1e-12, 1e-14, 1e-16, 1e-17, 1e-50):
        x, y = kernel(b_field)
        xe, ye = matrix(b_field)
        xh, yh = kernel_in_60_digits(b_field)
        err = max(abs(x - xe), abs(y - ye))
        control = b_field == 1e-2
        print(
            f"[soln B = {b_field:.0e} m^-1{' (control)' if control else ''}] x observed {x:.12e} m, expected {xe:.12e} "
            f"(error {x - xe:+.3e}); y observed {y:.12e}, expected {ye:.12e} (error {y - ye:+.3e}); "
            f"60-digit kernel minus expected: x {xh - xe:+.1e}, y {yh - ye:+.1e}"
        )
        controls_ok &= abs(xh - xe) < 1e-18 and abs(yh - ye) < 1e-18
        if control:
            controls_ok &= err < 1e-16
        elif b_field == 1e-8:
            defects += (
                err > 1e-12
            )  # small, but far above the 1e-16 conditioning of the B = 1e-2 control
        else:
            defects += err > 1e-9
    for label, run in (
        ("SolenoidTEAPOT in TEAPOT_Lattice", teapot_node),
        ("linac Solenoid in LinacAccLattice", linac_node),
    ):
        x, y = run(1e-17)
        xe, ye = matrix(1e-17)
        print(
            f"[{label}, B = 1e-17 m^-1] x observed {x:.12e} m, expected {xe:.12e}; y observed {y:.12e}, expected {ye:.12e}"
        )
        defects += max(abs(x - xe), abs(y - ye)) > 1e-9

    print("DEFECT REPRODUCED" if defects == 8 and controls_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

