---
title: "TRK-04: Axis-field RF gap phase solve stops one step before the gap centre"
hide:
  - toc
---

# TRK-04 — Axis-field RF gap phase solve stops one step before the gap centre

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`LinacRfGapNodes.py#L812-L861`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L812-L861)<br>[`LinacRfGapNodes.py#L722-L725`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L722-L725)<br>[`LinacFieldOverlappingNodes.py#L450-L453`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacFieldOverlappingNodes.py#L450-L453) |

`AxisFieldRF_Gap` finds the entrance phase that puts the reference at the cavity phase at the first gap's centre by stepping a test particle from the field start to the centre. When the accumulated position reaches exactly 0.0 in floating point, the loop exits one step early, so the solve matches the phase one step upstream and the gap runs off phase by about that step's transit phase. For a 2.5 MeV proton at 402.5 MHz, a −30° cavity phase and a field over ±10 mm in 1 mm steps, the reference crosses the centre at −23.41°; for ±20 mm at the default 10 mm step, at +36.03°. With 0.5 mm steps it crosses at −30.00°.

Run `python TRK-04.py`.

??? example "Reproducer: TRK-04.py"

    [Download TRK-04.py](TRK-04.py)

    ```python
    """TRK-04: Axis-field RF gap phase solve stops one step before the gap centre

    `AxisFieldRF_Gap` finds the entrance phase that puts the reference at the cavity phase at the first gap's centre by
    stepping a test particle from the field start to the centre. When the accumulated position reaches exactly 0.0 in
    floating point, the loop exits one step early, so the solve matches the phase one step upstream and the gap runs off
    phase by about that step's transit phase. For a 2.5 MeV proton at 402.5 MHz, a −30° cavity phase and a field over ±10 mm
    in 1 mm steps, the reference crosses the centre at −23.41°; for ±20 mm at the default 10 mm step, at +36.03°. With 0.5
    mm steps it crosses at −30.00°.

    The script builds a one-gap `AxisFieldRF_Gap` cavity with a cos² axis field normalized to unit integral, runs the lattice
    design pass, and integrates the reference with DOP853 through the continuous field from the entrance phase the solve
    stored, giving the RF phase at the gap centre, which should equal the cavity phase. It also prints the centre phase the
    gap reports and replays the solve's position loop in Python to show where it stops. The cases are ±10 mm in 1 mm steps
    and ±20 mm in the default 10 mm step; the control is ±10 mm in 0.5 mm steps, where the position sum lands just short of
    0.0 and the loop reaches the centre with one double-length last step.
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
    PHASE = -30.0  # deg, cavity phase = design phase at the first gap centre
    M = Bunch().mass()


    def shape(z, zmax):
        return math.cos(math.pi * z / (2 * zmax)) ** 2 / zmax  # integral over [-zmax, zmax] = 1


    def axis_field(zmax, n=401):
        fn = Function()
        for i in range(n):
            z = -zmax + 2 * zmax * i / (n - 1)
            fn.add(z, shape(z, zmax))
        fn.setConstStep(1)
        return fn


    def build(zmax, req_step):
        lat, seq, cav = LinacAccLattice("lat"), Sequence("seq"), RF_Cavity("cav")
        cav.setFrequency(F)
        cav.setPhase(math.radians(PHASE))
        cav.setAmp(1.0)
        base = BaseRF_Gap("gap")
        base.setParam("E0L", E0L)
        cav.addRF_GapNode(base)
        gap = AxisFieldRF_Gap(base)
        gap.setAxisFieldFunction(axis_field(zmax), req_step)
        cav.removeAllGapNodes()
        cav.addRF_GapNode(gap)
        seq.addNode(gap)
        seq.addRF_Cavity(cav)
        lat.addNode(gap)
        lat.initialize()
        b = Bunch()
        b.getSyncParticle().kinEnergy(K0)
        lat.trackDesignBunch(b)
        return gap, cav


    def centre_phase(zmax, phi_entrance):
        """RF phase when the reference reaches s = 0, from the field start at RF phase phi_entrance.

        State (t, p) with s as independent variable: dt/ds = W/(p c), dp/ds = (W/p) q E_z(s) cos(omega t + phi_entrance),
        W the total energy, p in GeV/c and E_z = E0L * shape in GV/m (unit charge).
        """
        p0 = math.sqrt(K0 * (K0 + 2 * M))

        def rhs(s, y):
            w = math.sqrt(y[1] ** 2 + M * M)
            return [
                w / (y[1] * C),
                (w / y[1]) * E0L * shape(s, zmax) * math.cos(OMEGA * y[0] + phi_entrance),
            ]

        sol = solve_ivp(rhs, (-zmax, 0.0), [0.0, p0], method="DOP853", rtol=1e-12, atol=[1e-24, 1e-17])
        return math.degrees(phi_entrance + OMEGA * sol.y[0, -1])


    def wrap(deg):
        return (deg + 180.0) % 360.0 - 180.0


    def solve_loop_end(z_min, step):
        # The position loop of AxisFieldRF_Gap.__calculate_first_part_phase (z_tolerance 1e-6 m)
        z_old, z, n = z_min, z_min + step, 0
        while z < 0.0:
            if (z + step) > 0.0:
                z = 0.0
                if abs(z - z_old) < 1e-6:
                    break
            n += 1
            z_old, z = z, z + step
        return z_old, n


    reproduced = True
    for zmax, req_step, is_control in (
        (0.010, 0.001, False),
        (0.020, 0.010, False),
        (0.010, 0.0005, True),
    ):
        gap, cav = build(zmax, req_step)
        step = gap.getZ_Step()
        end, n = solve_loop_end(-zmax, step)
        true_centre = wrap(centre_phase(zmax, cav.getFirstGapEntrancePhase()))
        reported = wrap(math.degrees(gap.getGapPhase()))
        print(
            f"field +-{zmax * 1e3:.0f} mm, step {step * 1e3:.2f} mm: solve stops at z = {end * 1e3:.3f} mm after {n} of "
            f"{round(zmax / step)} steps; centre phase {true_centre:.4f} deg (direct integration; gap reports {reported:.4f}), "
            f"expected {PHASE:.4f} deg (cavity phase)"
        )
        if is_control:
            reproduced &= abs(true_centre - PHASE) < 0.01
        else:
            reproduced &= abs(true_centre - PHASE) > 1.0
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

