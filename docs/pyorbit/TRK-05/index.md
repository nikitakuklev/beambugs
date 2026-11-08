---
title: "TRK-05: RingRFTEAPOT adds the synchronous energy gain to every particle's dE as well as to the reference"
hide:
  - toc
---

# TRK-05 — `RingRFTEAPOT` adds the synchronous energy gain to every particle's dE as well as to the reference

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapotbase.cc#L1545-L1566`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L1545-L1566)<br>[`Harmonic_Cav.cc#L108-L110`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/RFCavities/Harmonic_Cav.cc#L108-L110) |

`RingRFTEAPOT`, with voltage V, synchronous phase φs and harmonic number h in a ring of length C, raises the reference kinetic energy by q·V·sin φs and adds q·V·sin(2πhz/C + φs) to each particle's dE, but dE is measured from the reference, so the synchronous gain is counted twice. A particle on the synchronous orbit (z = dE = 0) must keep dE = 0; it instead gains q·V·sin φs on every pass and slides away from the bucket centre. With V = 10⁻³ GV and φs = 0.3 rad in a 10 m ring, it has dE = 2.955×10⁻⁴ GeV after one pass and 6.284×10⁻³ GeV with z = 9.029×10⁻² m after twenty. Stationary buckets (φs = 0) are unaffected.

Run `python TRK-05.py`.

??? example "Reproducer: TRK-05.py"

    [Download TRK-05.py](TRK-05.py)

    ```python
    """TRK-05: `RingRFTEAPOT` adds the synchronous energy gain to every particle's dE as well as to the reference

    `RingRFTEAPOT`, with voltage V, synchronous phase φs and harmonic number h in a ring of length C, raises the reference
    kinetic energy by q·V·sin φs and adds q·V·sin(2πhz/C + φs) to each particle's dE, but dE is measured from the reference,
    so the synchronous gain is counted twice. A particle on the synchronous orbit (z = dE = 0) must keep dE = 0; it instead
    gains q·V·sin φs on every pass and slides away from the bucket centre. With V = 10⁻³ GV and φs = 0.3 rad in a 10 m ring,
    it has dE = 2.955×10⁻⁴ GeV after one pass and 6.284×10⁻³ GeV with z = 9.029×10⁻² m after twenty. Stationary buckets (φs
    = 0) are unaffected.

    The script builds a 10 m ring of a drift and one `RingRFTEAPOT` with V = 10⁻³ GV, tracks a proton at z = dE = 0 for
    twenty passes and prints, after passes 1, 3 and 20, the reference energy gain and the particle's dE and z. The expected
    dE and z are 0, because the particle and the reference receive the same energy gain at z = 0. The control is the same ring with φs = 0, a stationary bucket.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.teapot import DriftTEAPOT, RingRFTEAPOT, TEAPOT_Lattice

    V = 1e-3  # GV
    K0 = 1.0  # GeV


    def run(phi_s, passes=(1, 3, 20)):
        lat = TEAPOT_Lattice("ring")
        rf = RingRFTEAPOT("rf")
        rf.addRF(1, V, phi_s)  # harmonic 1
        lat.addNode(DriftTEAPOT("d", 10.0))
        lat.addNode(rf)
        lat.initialize()
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(K0)
        b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        rows = []
        for n in range(1, max(passes) + 1):
            lat.trackBunch(b)
            if n in passes:
                k_ref = b.getSyncParticle().kinEnergy()
                # A particle at z = 0 gains q*V*sin(phi_s) per pass, the same as the reference (K_ref = K0 + n*q*V*sin(phi_s)),
                # so its deviation dE = K - K_ref stays 0 and so does z.
                rows.append((n, k_ref - K0, b.dE(0), b.z(0)))
        return rows


    reproduced = True
    for phi_s, is_control in ((0.3, False), (0.0, True)):
        for n, gain, de, z in run(phi_s):
            print(
                f"phi_s={phi_s} rad, pass {n:2d}: reference gain {gain:.6e} GeV (n q V sin(phi_s) = "
                f"{n * V * math.sin(phi_s):.6e}); dE observed {de:.6e} GeV, expected 0; z observed {z:.6e} m, expected 0"
            )
            reproduced &= (abs(de) < 1e-15 and abs(z) < 1e-15) if is_control else abs(de) > 1e-6
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

