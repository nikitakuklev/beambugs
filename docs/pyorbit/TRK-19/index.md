---
title: "TRK-19: Linac RF gaps return NaN at the RF_Cavity default frequency of zero"
hide:
  - toc
---

# TRK-19 — Linac RF gaps return NaN at the `RF_Cavity` default frequency of zero

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`LinacAccLatticeLib.py#L370-L372`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccLatticeLib.py#L370-L372)<br>[`LinacRfGapNodes.py#L241-L250`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L241-L250)<br>[`#L283-L291`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L283-L291)<br>[`RfGapTTF.cc#L71-L74`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapTTF.cc#L71-L74)<br>[`RfGapTTF.cc#L104-L107`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapTTF.cc#L104-L107)<br>[`BaseRfGap.cc#L100`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/BaseRfGap.cc#L100-L100) |

A linac `RF_Cavity` starts with frequency 0.0 Hz and amplitude 1.0, and nothing checks the frequency, so a hand-built lattice that never calls `setFrequency` tracks its gaps as active RF gaps at zero frequency. The default `RfGapTTF` divides by the frequency in its phase correction and in z = phase/κ, κ = 2πf/(βc): with E₀L = 0.1 MeV at 2.5 MeV the reference time and every particle's z become NaN, and the NaN time then makes x′, dE and z NaN after every later cavity, including a correctly set 402.5 MHz one. `BaseRfGap` gives NaN x′ and y′ for off-axis particles, the three-point kernels NaN everywhere, and the axis-field gap's design pass raises a `ValueError`. The same lattice at 1 mHz tracks finitely.

Run `python TRK-19.py`.

??? example "Reproducer: TRK-19.py"

    [Download TRK-19.py](TRK-19.py)

    ```python
    """TRK-19: Linac RF gaps return NaN at the `RF_Cavity` default frequency of zero

    A linac `RF_Cavity` starts with frequency 0.0 Hz and amplitude 1.0, and nothing checks the frequency, so a hand-built lattice that never calls `setFrequency` tracks its gaps as active RF gaps at zero frequency. The default `RfGapTTF` divides by the frequency in its phase correction and in z = phase/κ, κ = 2πf/(βc): with E₀L = 0.1 MeV at 2.5 MeV the reference time and every particle's z become NaN, and the NaN time then makes x′, dE and z NaN after every later cavity, including a correctly set 402.5 MHz one. `BaseRfGap` gives NaN x′ and y′ for off-axis particles, the three-point kernels NaN everywhere, and the axis-field gap's design pass raises a `ValueError`. The same lattice at 1 mHz tracks finitely.

    The script builds a linac lattice of one `BaseRF_Gap` between zero-length drifts in an `RF_Cavity` left at its default frequency, runs the design pass and tracks an off-axis, off-energy particle and a reference particle, then repeats with a 402.5 MHz cavity after it. It compares z, x′ and dE with the f → 0 limit of the gap map, which the same lattice at 1 mHz reproduces, and calls the `BaseRfGap` kernel at 0 Hz and 1 mHz and the `RfGapThreePointTTF` kernel at 0 Hz and 1 kHz directly. Last, it runs the design pass of an `AxisFieldRF_Gap` cavity left at the default frequency. The runs at 1 mHz, 1 kHz and 402.5 MHz are the controls.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.core.linac import BaseRfGap, RfGapThreePointTTF
    from orbit.core.orbit_utils import Function
    from orbit.py_linac.lattice import (
        AxisFieldRF_Gap,
        BaseRF_Gap,
        Drift,
        LinacAccLattice,
        RF_Cavity,
        Sequence,
    )

    MASS = 0.938272013
    EKIN = 0.0025  # GeV
    E0TL = 1.0e-4  # GeV
    PHASE = math.radians(-30.0)
    PART = (1e-3, 1e-3, 0.5e-3, 2e-3, 1e-3, 1e-5)  # x, xp, y, yp, z, dE


    def make_bunch():
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(EKIN)
        b.addParticle(*PART)
        b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return b


    def lattice(freq, second=False):
        lat = LinacAccLattice("lat")
        seq = Sequence("seq")
        seq.setLinacAccLattice(lat)
        lat.addSequence(seq)
        cav = RF_Cavity("cav")
        if freq is not None:
            cav.setFrequency(freq)
        gap = BaseRF_Gap("gap")
        gap.setParam("E0TL", E0TL)
        gap.setParam("E0L", E0TL)  # default TTF polynomials: T = 1, S = T' = S' = 0
        cav.addRF_GapNode(gap)
        cav.setPhase(PHASE)
        seq.addRF_Cavity(cav)
        nodes = [Drift("d1"), gap, Drift("d2")]
        if second:
            # a following, correctly configured 402.5 MHz cavity
            cav2 = RF_Cavity("cav2")
            cav2.setFrequency(402.5e6)
            gap2 = BaseRF_Gap("gap2")
            gap2.setParam("E0TL", E0TL)
            gap2.setParam("E0L", E0TL)
            cav2.addRF_GapNode(gap2)
            cav2.setPhase(PHASE)
            seq.addRF_Cavity(cav2)
            nodes += [gap2, Drift("d3")]
        for node in nodes:
            if isinstance(node, Drift):
                node.setLength(0.0)  # zero-length drifts isolate the gap maps
            seq.addNode(node)
            lat.addNode(node)
        lat.initialize()
        return lat, cav


    def track(freq, second=False):
        lat, cav = lattice(freq, second)
        b = make_bunch()
        lat.trackDesignBunch(b)
        lat.trackBunch(b)
        sp = b.getSyncParticle()
        return cav.getFrequency(), b, sp.time(), sp.kinEnergy()


    def dc_limit(ek_out):
        # kappa -> 0 limit of RfGapTTF with T = 1, S = T' = S' = 0: phase_rf -> phase, I0 -> 1,
        # I1(Kr r)/r -> 0, so dE gains qE0L cos(phase) like the reference (net 0), z = phase/kappa
        # scales by beta_out/beta_in, x' by (beta*gamma)_in/(beta*gamma)_out, delta_time -> 0.
        def bg(ek):
            p = math.sqrt(ek * (ek + 2 * MASS))
            e = math.sqrt(p * p + MASS * MASS)
            return p / e, e / MASS

        b_in, g_in = bg(EKIN)
        b_out, g_out = bg(ek_out)
        return (PART[4] * b_out / b_in, PART[1] * (b_in * g_in) / (b_out * g_out), PART[5])


    def axis_field_design(freq):
        # one-gap AxisFieldRF_Gap cavity with a uniform field over +-10 mm in 0.5 mm steps
        lat, seq, cav = LinacAccLattice("lat"), Sequence("seq"), RF_Cavity("cav")
        if freq is not None:
            cav.setFrequency(freq)
        cav.setPhase(PHASE)
        cav.setAmp(1.0)
        base = BaseRF_Gap("gap")
        base.setParam("E0L", E0TL)
        cav.addRF_GapNode(base)
        gap = AxisFieldRF_Gap(base)
        fn = Function()
        fn.add(-0.01, 50.0)
        fn.add(0.01, 50.0)
        fn.setConstStep(1)
        gap.setAxisFieldFunction(fn, 0.0005)
        cav.removeAllGapNodes()
        cav.addRF_GapNode(gap)
        seq.addNode(gap)
        seq.addRF_Cavity(cav)
        lat.addNode(gap)
        lat.initialize()
        b = make_bunch()
        try:
            lat.trackDesignBunch(b)
        except ValueError as exc:
            return f"ValueError: {exc}"
        lat.trackBunch(b)
        return (
            f"design pass completes; tracked reference energy {b.getSyncParticle().kinEnergy():.9e} GeV"
        )


    f_default, b0, t0, ek0 = track(None)
    _, b1, t1, ek1 = track(1.0e-3)
    _, b2, t2, _ = track(402.5e6)
    z_exp, xp_exp, de_exp = dc_limit(ek0)
    nan = math.isnan
    print(
        f"RF_Cavity default frequency: {f_default} Hz; reference kinetic energy after the gap {ek0:.12e} GeV "
        f"(expected Ek + E0L cos(phase) = {EKIN + E0TL * math.cos(PHASE):.12e})"
    )
    print(
        f"[RfGapTTF lattice, default 0 Hz] z observed {b0.z(0)!r}, expected {z_exp:.12e} (z beta_out/beta_in); "
        f"xp observed {b0.xp(0)!r}, expected {xp_exp:.12e}; dE observed {b0.dE(0)!r}, expected {de_exp:.12e}; "
        f"reference particle z observed {b0.z(1)!r}, expected 0; reference time observed {t0!r}, expected 0 s"
    )
    print(
        f"[RfGapTTF lattice, 1 mHz (control)] z {b1.z(0):.12e} (relative to the limit {(b1.z(0) - z_exp) / z_exp:+.1e}); "
        f"xp {b1.xp(0):.12e} (relative {(b1.xp(0) - xp_exp) / xp_exp:+.1e}); dE {b1.dE(0):.12e}; time {t1:.3e} s"
    )
    print(
        f"[RfGapTTF lattice, 402.5 MHz (control)] z {b2.z(0):.12e}, xp {b2.xp(0):.12e}, dE {b2.dE(0):.12e}, time {t2:.3e} s"
    )
    _, b3, t3, ek3 = track(None, second=True)
    _, b4, t4, ek4 = track(1.0e-3, second=True)
    print(
        f"[default-frequency cavity, then a 402.5 MHz cavity] xp observed {b3.xp(0)!r}, dE {b3.dE(0)!r}, z {b3.z(0)!r}, "
        f"reference time {t3!r}, reference energy {ek3!r}; expected the values with a 1 mHz first cavity: xp "
        f"{b4.xp(0):.12e}, dE {b4.dE(0):.12e}, z {b4.z(0):.12e}, time {t4:.6e} s, energy {ek4:.12e} GeV"
    )
    kernels = {}
    for f in (0.0, 1.0e-3):
        b = make_bunch()
        BaseRfGap().trackBunch(b, f, E0TL, PHASE)
        kernels["base", f] = (b.xp(0), b.yp(0), b.z(0))
    # The three-point kernel's transit-time derivative loses precision below about 1 Hz for a 10 mm step,
    # so its finite control runs at 1 kHz.
    for f in (0.0, 1.0e3):
        b = make_bunch()
        # one 10 mm step with fields of 9, 10 and 9 MV/m at its start, centre and end
        RfGapThreePointTTF().trackBunch(b, 0.005, 9e6, 1.0e7, 9e6, f, PHASE)
        kernels["3pt", f] = (b.xp(0), b.z(0), b.dE(0), b.getSyncParticle().kinEnergy())
    for f in (0.0, 1.0e-3):
        xp, yp, z = kernels["base", f]
        print(
            f"[BaseRfGap kernel, {f} Hz{' (control)' if f else ''}] xp observed {xp!r}, yp observed {yp!r}, z {z!r}; "
            f"expected xp {xp_exp:.12e} (limit: 2 I1(kr r)/(kr r) -> 1 and the kick vanishes with kappa)"
        )
    for f in (0.0, 1.0e3):
        xp, z, de, ek = kernels["3pt", f]
        print(
            f"[RfGapThreePointTTF kernel, {f} Hz{' (control)' if f else ''}] xp {xp!r}, z {z!r}, dE {de!r}, "
            f"reference energy {ek!r}; expected finite values"
        )
    axis_default, axis_1khz = axis_field_design(None), axis_field_design(1.0e3)
    print(
        f"[AxisFieldRF_Gap design pass, default 0 Hz] {axis_default}; expected the design pass to complete"
    )
    print(f"[AxisFieldRF_Gap design pass, 1 kHz (control)] {axis_1khz}")

    controls_ok = (
        all(
            math.isfinite(v)
            for v in (b1.z(0), b1.xp(0), t1, b2.z(0), b2.xp(0), t2, b4.dE(0), b4.xp(0), t4)
        )
        and abs((b1.z(0) - z_exp) / z_exp) < 1e-11
        and abs((b1.xp(0) - xp_exp) / xp_exp) < 1e-11
        and all(math.isfinite(v) for v in kernels["base", 1.0e-3] + kernels["3pt", 1.0e3])
        and axis_1khz.startswith("design pass completes")
    )
    reproduced = (
        nan(b0.z(0))
        and nan(t0)
        and all(nan(v) for v in (b3.xp(0), b3.dE(0), b3.z(0), t3))
        and nan(kernels["base", 0.0][0])
        and all(nan(v) for v in kernels["3pt", 0.0])
        and axis_default.startswith("ValueError")
    )
    print("DEFECT REPRODUCED" if reproduced and controls_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed<br><small>three-point kernels and axis-field gap</small> | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

