---
title: "TRK-08: The axis-field design pass tracks its first part with the previous design amplitude"
hide:
  - toc
---

# TRK-08 — The axis-field design pass tracks its first part with the previous design amplitude

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`LinacRfGapNodes.py#L714-L746`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L714-L746)<br>[`LinacRfGapNodes.py#L796`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L796-L796)<br>[`LinacAccLatticeLib.py#L374`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccLatticeLib.py#L374-L374)<br>[`LinacFieldOverlappingNodes.py#L442-L458`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacFieldOverlappingNodes.py#L442-L458)<br>[`LinacRfGapNodes.py#L268-L283`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L268-L283) |

In `AxisFieldRF_Gap.trackDesign`, the first part of a cavity's first gap reads the cavity's design amplitude before the same call copies the current amplitude into it. The first design pass of a cavity whose amplitude is not 1.0, and the first pass after any amplitude change, therefore track that part with the old value (1.0 for a new cavity), while the phase solve and all later parts use the new one. For a 2.5 MeV proton at 402.5 MHz, a −30° cavity phase, field integral E₀L = 0.1 MeV and a uniform field over ±10 mm in 0.5 mm steps at amplitude 0.5, the first pass gains 34.18 keV and the next ones 34.28 keV, the value direct integration through the field gives.

Run `python TRK-08.py`.

??? example "Reproducer: TRK-08.py"

    [Download TRK-08.py](TRK-08.py)

    ```python
    """TRK-08: The axis-field design pass tracks its first part with the previous design amplitude

    In `AxisFieldRF_Gap.trackDesign`, the first part of a cavity's first gap reads the cavity's design amplitude before the
    same call copies the current amplitude into it. The first design pass of a cavity whose amplitude is not 1.0, and the
    first pass after any amplitude change, therefore track that part with the old value (1.0 for a new cavity), while the
    phase solve and all later parts use the new one. For a 2.5 MeV proton at 402.5 MHz, a −30° cavity phase, field integral
    E₀L = 0.1 MeV and a uniform field over ±10 mm in 0.5 mm steps at amplitude 0.5, the first pass gains 34.18 keV and the
    next ones 34.28 keV, the value direct integration through the field gives.

    The script builds a one-gap `AxisFieldRF_Gap` cavity with a uniform axis field over ±10 mm in 0.5 mm steps, sets the
    cavity amplitude to 0.5 after initialization and runs the lattice design pass three times with the same input bunch. It
    compares each pass's reference energy gain with a DOP853 integration of the reference through the continuous field at
    amplitude 0.5, started at the entrance phase the solve stored. The control is the same cavity at amplitude 1.0, the
    initial design amplitude, where the first pass already agrees with the later ones.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.core.orbit_utils import Function
    from orbit.py_linac.lattice import AxisFieldRF_Gap, BaseRF_Gap, LinacAccLattice, RF_Cavity, Sequence
    from scipy.integrate import solve_ivp

    C = 2.99792458e8
    F = 402.5e6
    OMEGA = 2 * math.pi * F
    K0 = 2.5e-3  # GeV
    E0L = 1e-4  # GeV
    ZMAX = 0.010  # m
    M = Bunch().mass()


    def build(amp):
        lat, seq, cav = LinacAccLattice("lat"), Sequence("seq"), RF_Cavity("cav")
        cav.setFrequency(F)
        cav.setPhase(math.radians(-30.0))
        cav.setAmp(1.0)
        base = BaseRF_Gap("gap")
        base.setParam("E0L", E0L)
        cav.addRF_GapNode(base)
        gap = AxisFieldRF_Gap(base)
        fn = Function()
        fn.add(-ZMAX, 1 / (2 * ZMAX))  # uniform field of unit integral over [-ZMAX, ZMAX]
        fn.add(ZMAX, 1 / (2 * ZMAX))
        fn.setConstStep(1)
        gap.setAxisFieldFunction(fn, 0.0005)
        cav.removeAllGapNodes()
        cav.addRF_GapNode(gap)
        seq.addNode(gap)
        seq.addRF_Cavity(cav)
        lat.addNode(gap)
        lat.initialize()
        cav.setAmp(
            amp
        )  # the cavity's design amplitude keeps its initial value 1.0 until a design pass copies amp
        return lat, cav


    def design_gain(lat):
        b = Bunch()
        b.getSyncParticle().kinEnergy(K0)
        return (lat.trackDesignBunch(b).getSyncParticle().kinEnergy() - K0) * 1e6  # keV


    def integrated_gain(phi_entrance, amp):
        """Reference gain through E_z = amp*E0L/(2*ZMAX) GV/m on [-ZMAX, ZMAX], entering at RF phase phi_entrance.

        State (t, p) with s as independent variable: dt/ds = W/(p c), dp/ds = (W/p) q E_z cos(omega t + phi_entrance).
        """
        p0 = math.sqrt(K0 * (K0 + 2 * M))

        def rhs(s, y):
            w = math.sqrt(y[1] ** 2 + M * M)
            return [
                w / (y[1] * C),
                (w / y[1]) * amp * E0L / (2 * ZMAX) * math.cos(OMEGA * y[0] + phi_entrance),
            ]

        sol = solve_ivp(rhs, (-ZMAX, ZMAX), [0.0, p0], method="DOP853", rtol=1e-12, atol=[1e-24, 1e-17])
        return (math.sqrt(sol.y[1, -1] ** 2 + M * M) - M - K0) * 1e6


    reproduced = True
    for amp, is_control in ((0.5, False), (1.0, True)):
        lat, cav = build(amp)
        gains = [design_gain(lat) for _ in range(3)]
        expected = integrated_gain(cav.getFirstGapEntrancePhase(), amp)
        print(
            f"amplitude {amp}: design pass gains {gains[0]:.4f}, {gains[1]:.4f}, {gains[2]:.4f} keV; "
            f"expected {expected:.4f} keV each (direct integration at amplitude {amp}); "
            f"first - expected {(gains[0] - expected) * 1e3:+.2f} eV"
        )
        # The three-point steps differ from the continuous field by under 0.1 eV here, the stale amplitude by about 100 eV
        if is_control:
            reproduced &= all(abs(g - expected) < 1e-2 for g in gains)
        else:
            reproduced &= abs(gains[0] - expected) > 1e-2 and all(
                abs(g - expected) < 1e-2 for g in gains[1:]
            )
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed<br><small class="bb-scope">first design pass of a new cavity</small> | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

