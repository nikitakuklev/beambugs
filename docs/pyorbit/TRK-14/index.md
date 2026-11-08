---
title: "TRK-14: Dual_Harmonic_Cav evaluates the second harmonic at the fundamental's phase, so RFPhase2 only adds a constant to every particle's energy"
hide:
  - toc
---

# TRK-14 — `Dual_Harmonic_Cav` evaluates the second harmonic at the fundamental's phase, so `RFPhase2` only adds a constant to every particle's energy

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`Dual_Harmonic_Cav.cc#L127-L129`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/RFCavities/Dual_Harmonic_Cav.cc#L127-L129)<br>[`RFNode.py#L118-L126`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/rf_cavities/RFNode.py#L118-L126)<br>[`wrap_Dual_Harmonic_Cav.cc#L57-L66`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/RFCavities/wrap_Dual_Harmonic_Cav.cc#L57-L66) |

`Dual_Harmonic_RFNode` and its kernel `Dual_Harmonic_Cav`, with fundamental voltage V, voltage ratio r, harmonic ratio n and phases φ₁ and φ₂, add q·V[sin φ − sin φ₁ − r(sin(φ₁ + n(φ − φ₁)) − sin φ₂)] to dE, φ being the particle's RF phase. Subtracting each harmonic's value at the synchronous phase φ₁ requires the second-harmonic wave sin(φ₂ + n(φ − φ₁)); with φ₁ in its place, φ₂ only adds q·V·r·sin φ₂ to every particle, and for φ₁ ≠ φ₂ the synchronous particle gains q·V·r(sin φ₂ − sin φ₁) per pass. For 1 GeV protons in a 100 m drift ring at harmonic number 1, V = 1 MV, n = 2, r = 0.25, φ₁ = 0 and φ₂ = 180°, the synchrotron tune is 0.003545, that of φ₂ = 0, instead of 0.006140.

Run `python TRK-14.py`.

??? example "Reproducer: TRK-14.py"

    [Download TRK-14.py](TRK-14.py)

    ```python
    """TRK-14: `Dual_Harmonic_Cav` evaluates the second harmonic at the fundamental's phase, so `RFPhase2` only adds a constant to every particle's energy

    `Dual_Harmonic_RFNode` and its kernel `Dual_Harmonic_Cav`, with fundamental voltage V, voltage ratio r, harmonic ratio n and phases φ₁ and φ₂, add q·V[sin φ − sin φ₁ − r(sin(φ₁ + n(φ − φ₁)) − sin φ₂)] to dE, φ being the particle's RF phase. Subtracting each harmonic's value at the synchronous phase φ₁ requires the second-harmonic wave sin(φ₂ + n(φ − φ₁)); with φ₁ in its place, φ₂ only adds q·V·r·sin φ₂ to every particle, and for φ₁ ≠ φ₂ the synchronous particle gains q·V·r(sin φ₂ − sin φ₁) per pass. For 1 GeV protons in a 100 m drift ring at harmonic number 1, V = 1 MV, n = 2, r = 0.25, φ₁ = 0 and φ₂ = 180°, the synchrotron tune is 0.003545, that of φ₂ = 0, instead of 0.006140.

    The script tracks 1 GeV protons once through `Dual_Harmonic_Cav` with h = 1, n = 2 in a 100 m ring for several (φ₁, φ₂) pairs and compares each particle's energy change with the dual-harmonic kick whose second harmonic has phase φ₂ at the synchronous particle. It then measures the synchrotron tune of a ring of the cavity and a 100 m TEAPOT drift from the one-turn map by central differences and compares it with the tune of that kick's restoring slope. The controls are φ₁ = φ₂, the setting of the upstream test, and r = 0 and r = −0.25 at φ₂ = 0, where the code and the formula coincide.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core import teapot_base as TPB
    from orbit.core.bunch import Bunch
    from orbit.core.rfcavities import Dual_Harmonic_Cav

    C = 100.0  # ring length, m
    ZTOPHI = 2.0 * math.pi / C
    H, N = 1.0, 2.0  # harmonic number, harmonic ratio
    MASS, KE = 0.938272088, 1.0  # GeV


    def bunch(phis):
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(KE)
        for phi in phis:
            # the cavity's RF phase is -ZtoPhi*h*z, so a particle at RF phase phi sits at z = -phi/(ZtoPhi*h)
            b.addParticle(0.0, 0.0, 0.0, 0.0, -phi / (ZTOPHI * H), 0.0)
        return b


    def expected_kick(v, r, phi, phi1, phi2):
        # Kick relative to the synchronous particle at phi = phi1: each harmonic's value there is subtracted,
        # sin(phi1) for the fundamental and sin(phi2) for the second harmonic, whose phase at phi1 is phi2.
        return v * (
            math.sin(phi) - math.sin(phi1) - r * (math.sin(phi2 + N * (phi - phi1)) - math.sin(phi2))
        )


    defects, controls_ok = 0, True

    # ---------------------------------------------------------------- single-pass kicks
    V, R = 1e-5, 0.5  # GV, voltage ratio
    PHASES = (-0.3, 0.0, 0.3)  # rad
    for label, p1, p2, phis, control in (
        ("phi1 = 0, phi2 = 0 (control)", 0.0, 0.0, PHASES, True),
        ("phi1 = 0, phi2 = 180 deg", 0.0, 180.0, PHASES, False),
        ("phi1 = 0, phi2 = 90 deg", 0.0, 90.0, PHASES, False),
        ("phi1 = 20, phi2 = 40 deg, synchronous particle", 20.0, 40.0, (math.radians(20.0),), False),
    ):
        b = bunch(phis)
        Dual_Harmonic_Cav(ZTOPHI, H, N, V, R, p1, p2).trackBunch(b)
        case_bad = False
        for i, phi in enumerate(phis):
            exp = expected_kick(V, R, phi, math.radians(p1), math.radians(p2))
            bad = abs(b.dE(i) - exp) > 1e-12 * V
            case_bad |= bad
            print(
                f"[kick, r = {R}, {label}] phi {phi:+.4f} rad: dE observed {b.dE(i) / V:+.6e} q V, "
                f"expected {exp / V:+.6e} q V"
            )
        if control:
            controls_ok &= not case_bad
        else:
            defects += case_bad

    # ---------------------------------------------------------------- synchrotron tune of a drift ring
    V2 = 1e-3  # GV
    mom = math.sqrt((MASS + KE) ** 2 - MASS**2)
    beta, gamma = mom / (MASS + KE), (MASS + KE) / MASS
    A = C / (
        gamma**2 * mom * beta
    )  # TEAPOT drift: dz/d(dE) over the ring, z += L dE/(p0 beta0 gamma^2)


    def one_turn(r, p1, p2, z, de):
        b = bunch(())
        b.addParticle(0.0, 0.0, 0.0, 0.0, z, de)
        Dual_Harmonic_Cav(ZTOPHI, H, N, V2, r, p1, p2).trackBunch(b)
        TPB.drift(b, C)
        return b.z(0), b.dE(0)


    def tune_observed(r, p1, p2):
        hz, he = 1e-5, 1e-9
        zp, _ = one_turn(r, p1, p2, hz, 0.0)
        zm, _ = one_turn(r, p1, p2, -hz, 0.0)
        _, ep = one_turn(r, p1, p2, 0.0, he)
        _, em = one_turn(r, p1, p2, 0.0, -he)
        trace = (zp - zm) / (2 * hz) + (ep - em) / (2 * he)
        return math.acos(trace / 2) / (2 * math.pi)


    for label, r, p1, p2, control in (
        ("r = 0", 0.0, 0.0, 0.0, True),
        ("r = 0.25, phi1 = phi2 = 0", 0.25, 0.0, 0.0, True),
        ("r = -0.25, phi1 = phi2 = 0", -0.25, 0.0, 0.0, True),
        ("r = 0.25, phi1 = 0, phi2 = 180 deg", 0.25, 0.0, 180.0, False),
        ("r = 0.25, phi1 = 0, phi2 = 90 deg", 0.25, 0.0, 90.0, False),
    ):
        # Restoring slope of the expected kick at phi = phi1: g = cos(phi1) - r n cos(phi2); with
        # d(phi)/dz = -ZtoPhi*h the kick is dE -= K z, K = q V ZtoPhi h g, and the cavity-then-drift map
        # has trace 2 - A K, so cos(2 pi nu) = 1 - A K/2.
        g = math.cos(math.radians(p1)) - r * N * math.cos(math.radians(p2))
        nu_exp = math.acos(1 - A * V2 * ZTOPHI * H * g / 2) / (2 * math.pi)
        nu = tune_observed(r, p1, p2)
        print(
            f"[tune, {label}{' (control)' if control else ''}] synchrotron tune observed {nu:.9f}, expected {nu_exp:.9f} "
            f"(slope g = {g:+.3f})"
        )
        if control:
            controls_ok &= abs(nu - nu_exp) < 1e-8
        else:
            defects += abs(nu - nu_exp) > 1e-6

    print("DEFECT REPRODUCED" if defects == 5 and controls_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

