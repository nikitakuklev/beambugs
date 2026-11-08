---
title: "TRK-15: SynchPartRedefinitionZdE.shift_dE and center_dE change the reference speed without rescaling z, which the default RF gap kernels read as an arrival time"
hide:
  - toc
---

# TRK-15 — `SynchPartRedefinitionZdE.shift_dE` and `center_dE` change the reference speed without rescaling z, which the default RF gap kernels read as an arrival time

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`SynchPartRedefinitionZdE.cc#L90-L127`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SynchPartRedefinition/SynchPartRedefinitionZdE.cc#L90-L127)<br>[`BaseRfGap.cc#L71-L91`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/BaseRfGap.cc#L71-L91)<br>[`RfGapTTF.cc#L58-L107`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapTTF.cc#L58-L107)<br>[`BaseRfGap_slow.cc#L106`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/BaseRfGap_slow.cc#L106-L106) |

`SynchPartRedefinitionZdE.shift_dE(bunch, ΔK)` and `center_dE(bunch)` move the reference kinetic energy by ΔK, keeping each particle's energy and transverse momentum, but leave z unchanged while the reference speed β₀ changes. The default linac RF gap kernels read z as −β₀cΔt, with Δt the arrival time after the reference, so after the shift every particle's Δt is scaled by β₀,old/β₀,new and the next gap gives it the wrong energy; z·β₀,new/β₀,old is required. A 50 keV shift of a 2.5 MeV proton bunch shortens Δt by 0.9813%, and a following 402.5 MHz gap with E₀TL = 0.1 MeV at −90° gives a particle 5 mm ahead 476.4 eV more energy than without the shift. The per-particle kernels of the linac tracker are unaffected.

Run `python TRK-15.py`.

??? example "Reproducer: TRK-15.py"

    [Download TRK-15.py](TRK-15.py)

    ```python
    """TRK-15: `SynchPartRedefinitionZdE.shift_dE` and `center_dE` change the reference speed without rescaling z, which the default RF gap kernels read as an arrival time

    `SynchPartRedefinitionZdE.shift_dE(bunch, ΔK)` and `center_dE(bunch)` move the reference kinetic energy by ΔK, keeping each particle's energy and transverse momentum, but leave z unchanged while the reference speed β₀ changes. The default linac RF gap kernels read z as −β₀cΔt, with Δt the arrival time after the reference, so after the shift every particle's Δt is scaled by β₀,old/β₀,new and the next gap gives it the wrong energy; z·β₀,new/β₀,old is required. A 50 keV shift of a 2.5 MeV proton bunch shortens Δt by 0.9813%, and a following 402.5 MHz gap with E₀TL = 0.1 MeV at −90° gives a particle 5 mm ahead 476.4 eV more energy than without the shift. The per-particle kernels of the linac tracker are unaffected.

    The script redefines the reference of a 2.5 MeV proton bunch by +50 keV with `shift_dE` and with `center_dE` (a bunch whose mean energy deviation is 50 keV) and compares each particle's kinetic energy, transverse momentum and arrival time before and after; energy and transverse momentum are controls that pass, as does a zero shift. It then passes a particle 5 mm ahead with dE = +50 keV through a thin gap with and without the shift and compares its exit kinetic energy for the default `RfGapTTF` kernel and `BaseRfGap`, for the same kernels after the shift plus z·β₀,new/β₀,old, and for the linac tracker's `_slow` kernels, which take the phase from each particle's own speed.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch, SynchPartRedefinitionZdE
    from orbit.core.linac import BaseRfGap, BaseRfGap_slow, RfGapTTF, RfGapTTF_slow
    from orbit.core.orbit_utils import Polynomial

    C_LIGHT = 2.99792458e8
    MP = 0.938272013
    K0, DK, F, E0TL, PHI = 0.0025, 5.0e-5, 402.5e6, 1.0e-4, math.radians(-90.0)
    PARTS = [(0.0, 1.0e-3, 0.0, 0.0, 5.0e-3, DK), (0.0, -1.0e-3, 0.0, 0.0, -5.0e-3, DK)]


    def bunch_with(parts):
        b = Bunch()
        b.mass(MP)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(K0)
        for p in parts:
            b.addParticle(*p)
        return b


    def state(b, i):
        sp = b.getSyncParticle()
        # kinetic energy, p_x in GeV/c (xp = p_x/p0) and arrival time t_ref - z/(beta0 c), from z = -beta0 c dt
        return (
            sp.kinEnergy() + b.dE(i),
            sp.momentum() * b.xp(i),
            sp.time() - b.z(i) / (sp.beta() * C_LIGHT),
        )


    defects, controls_ok = 0, True
    for label, op, control in (
        ("shift_dE(+50 keV)", lambda b: SynchPartRedefinitionZdE().shift_dE(b, DK), False),
        ("center_dE, mean dE 50 keV", lambda b: SynchPartRedefinitionZdE().center_dE(b), False),
        ("shift_dE(0) (control)", lambda b: SynchPartRedefinitionZdE().shift_dE(b, 0.0), True),
    ):
        b = bunch_with(PARTS)
        before = [state(b, i) for i in range(2)]
        op(b)
        after = [state(b, i) for i in range(2)]
        for i in range(2):
            (k0, px0, t0), (k1, px1, t1) = before[i], after[i]
            # Redefining the reference at a fixed event changes no particle: energy, p_x and arrival time stay.
            same_kp = abs(k1 - k0) < 1e-15 and abs(px1 - px0) < 1e-15
            moved = abs(t1 - t0) > 1e-12 * abs(t0)
            print(
                f"[{label}] particle {i}: K observed {k1:.12e} GeV, expected {k0:.12e}; p_x observed {px1:.12e} GeV/c, "
                f"expected {px0:.12e}; arrival time observed {t1:.9e} s, expected {t0:.9e} s; "
                f"relative change {(t1 - t0) / t0:+.4e}"
            )
            controls_ok &= same_kp
            if control:
                controls_ok &= not moved
            else:
                defects += moved


    def constant(c):
        p = Polynomial(0)
        p.coefficient(0, c)
        return p


    class TTF:
        # RfGapTTF with the transit-time factors of BaseRF_Gap's defaults, T = 1 and S = T' = S' = 0
        def __init__(self, kernel):
            self.kernel = kernel

        def trackBunch(self, b, f, e0l, phi):
            self.kernel.trackBunch(
                b, f, e0l, phi, constant(1.0), constant(0.0), constant(0.0), constant(0.0)
            )


    def exit_energy(gap, dk, rescale):
        b = bunch_with([(0.0, 0.0, 0.0, 0.0, 5.0e-3, DK)])
        if dk:
            beta_old = b.getSyncParticle().beta()
            SynchPartRedefinitionZdE().shift_dE(b, dk)
            if rescale:
                b.z(0, b.z(0) * b.getSyncParticle().beta() / beta_old)
        gap.trackBunch(b, F, E0TL, PHI)
        return b.getSyncParticle().kinEnergy() + b.dE(0)


    # The same particle at the same event must leave the gap with the same kinetic energy whatever reference
    # was chosen at the entrance, so the expected difference from the unshifted run is 0.
    for label, gap, rescale, control in (
        ("RfGapTTF (default kernel)", TTF(RfGapTTF()), False, False),
        ("BaseRfGap", BaseRfGap(), False, False),
        ("RfGapTTF, shift plus z*beta_new/beta_old (control)", TTF(RfGapTTF()), True, True),
        ("BaseRfGap, shift plus z*beta_new/beta_old (control)", BaseRfGap(), True, True),
        ("RfGapTTF_slow (linac tracker, control)", TTF(RfGapTTF_slow()), False, True),
        ("BaseRfGap_slow (linac tracker, control)", BaseRfGap_slow(), False, True),
    ):
        k_ref = exit_energy(gap, 0.0, False)
        for dk in (1.0e-5, DK):
            diff = exit_energy(gap, dk, rescale) - k_ref
            print(
                f"[{label}, shift {dk * 1e6:.0f} keV] exit K minus unshifted exit K: observed {diff * 1e9:+.2f} eV, "
                f"expected 0"
            )
            if control:
                controls_ok &= abs(diff) < 1e-14
            else:
                defects += abs(diff) > 1e-9

    print("DEFECT REPRODUCED" if defects == 8 and controls_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed<br><small>default gap kernels; not the linac tracker</small> | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

