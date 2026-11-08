---
title: "TRK-09: The three-point RF gap kernels skip a step whose centre field is zero"
hide:
  - toc
---

# TRK-09 — The three-point RF gap kernels skip a step whose centre field is zero

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`RfGapThreePointTTF.cc#L44-L55`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapThreePointTTF.cc#L44-L55)<br>[`RfGapThreePointTTF_slow.cc#L48`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapThreePointTTF_slow.cc#L48-L48)<br>[`LinacRfGapNodes.py#L632-L650`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L632-L650) |

`RfGapThreePointTTF`, the kernel of `AxisFieldRF_Gap`, fits E(s) = E₀(1 + as + bs²) through the fields at a step's start, centre and end, and returns without any kick when the centre field E₀ is zero, even when the end fields are not. The quadratic through those three points has a nonzero integral, and the kernel's energy gain has a finite limit as E₀ → 0. For a 2.5 MeV proton at 402.5 MHz, φ = −30° and a 10 mm step, end fields of 0 and 1 MV/m, as at the onset of a field, give no energy change instead of 1.768 keV, and end fields of 1 MV/m give none instead of 2.602 keV. The per-particle kernel behaves the same.

Run `python TRK-09.py`.

??? example "Reproducer: TRK-09.py"

    [Download TRK-09.py](TRK-09.py)

    ```python
    """TRK-09: The three-point RF gap kernels skip a step whose centre field is zero

    `RfGapThreePointTTF`, the kernel of `AxisFieldRF_Gap`, fits E(s) = E₀(1 + as + bs²) through the fields at a step's
    start, centre and end, and returns without any kick when the centre field E₀ is zero, even when the end fields are not.
    The quadratic through those three points has a nonzero integral, and the kernel's energy gain has a finite limit as E₀ →
    0. For a 2.5 MeV proton at 402.5 MHz, φ = −30° and a 10 mm step, end fields of 0 and 1 MV/m, as at the onset of a field,
    give no energy change instead of 1.768 keV, and end fields of 1 MV/m give none instead of 2.602 keV. The per-particle
    kernel behaves the same.

    The script calls `RfGapThreePointTTF` and `RfGapThreePointTTF_slow` for a single 10 mm step with end fields of +1 and
    +1 MV/m, −1 and +1 MV/m, and 0 and +1 MV/m, and a centre field of exactly 0. The expected gain is q∫E(s)cos(φ + κs)ds over
    the quadratic through the three samples, with κ at the entrance velocity. The control is the same steps with centre
    fields of 10⁻³ V/m and 1 V/m, where the kernels reproduce that integral, showing it is continuous at E₀ = 0.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import math

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.linac import RfGapThreePointTTF, RfGapThreePointTTF_slow

    C = 2.99792458e8
    M = 0.938272013
    F = 402.5e6
    K0 = 2.5e-3
    PHI = math.radians(-30.0)
    DZ = 0.005  # half step, m


    def kernel_gain(kernel, em, e0, ep):
        b = Bunch()
        b.getSyncParticle().kinEnergy(K0)
        kernel.trackBunch(b, DZ, em, e0, ep, F, PHI)
        return (b.getSyncParticle().kinEnergy() - K0) * 1e6  # keV


    def quad_gain(em, e0, ep):
        # Quadratic through (-DZ, em), (0, e0), (DZ, ep); first-order gain int E(s) cos(phi + kappa s) ds, in keV
        beta = math.sqrt(1 - 1 / (1 + K0 / M) ** 2)
        kappa = 2 * math.pi * F / (C * beta)
        s = np.linspace(-DZ, DZ, 200001)
        e = e0 + (ep - em) / (2 * DZ) * s + (ep + em - 2 * e0) / (2 * DZ**2) * s**2
        return np.trapezoid(e * np.cos(PHI + kappa * s), s) * 1e-3  # V -> keV for unit charge


    reproduced = True
    for kernel in (RfGapThreePointTTF, RfGapThreePointTTF_slow):
        for em, ep in ((1e6, 1e6), (-1e6, 1e6), (0.0, 1e6)):
            for e0 in (0.0, 1e-3, 1.0):
                obs, exp = kernel_gain(kernel(), em, e0, ep), quad_gain(em, e0, ep)
                print(
                    f"{kernel.__name__:23s} Em {em:+.0e} V/m, E0 {e0:.0e} V/m, Ep {ep:+.0e} V/m: gain {obs:.4f} keV, "
                    f"expected {exp:.4f} keV (integral over the three-point quadratic)"
                )
                # E0 > 0 agrees within 0.2 eV: the kernel evaluates its TTFs at the midpoint velocity, the integral at the entrance
                reproduced &= abs(obs - exp) > 0.1 if e0 == 0.0 else abs(obs - exp) < 1e-3
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

