---
title: "TRK-07: The linac tracker scales dipole corrector and thick-kicker kicks by each particle's momentum ratio"
hide:
  - toc
---

# TRK-07 — The linac tracker scales dipole corrector and thick-kicker kicks by each particle's momentum ratio

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`linac_tracking.cc#L324-L371`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/tracking/linac_tracking.cc#L324-L371)<br>[`LinacAccNodes.py#L750-L751`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L750-L751)<br>[`LinacAccNodes.py#L795-L796`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L795-L796)<br>[`LinacAccNodes.py#L862-L865`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L862-L865)<br>[`linac_tracking.cc#L89-L104`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/tracking/linac_tracking.cc#L89-L104) |

With `setLinacTracker(True)`, linac dipole correctors and thick kickers call the linac `kick`, which multiplies the transverse kick by p₀/p_z, with p₀ the reference momentum and p_z the particle's longitudinal momentum. The linac tracker's transverse coordinates are pₓ/p₀ (its drift, quadrupole and RF gap maps all use this), and a field B_y over a length L changes pₓ by −qB_yL whatever the particle's momentum, so every particle must receive the node's kick −qB_yL/p₀ unchanged. A particle with relative momentum deviation δ instead receives (1 + δ)⁻¹ of it, 1.961% less at δ = +2%. The TEAPOT kick, used when the linac tracker is off, is exact.

Run `python TRK-07.py`.

??? example "Reproducer: TRK-07.py"

    [Download TRK-07.py](TRK-07.py)

    ```python
    """TRK-07: The linac tracker scales dipole corrector and thick-kicker kicks by each particle's momentum ratio

    With `setLinacTracker(True)`, linac dipole correctors and thick kickers call the linac `kick`, which multiplies the
    transverse kick by p₀/p_z, with p₀ the reference momentum and p_z the particle's longitudinal momentum. The linac
    tracker's transverse coordinates are pₓ/p₀ (its drift, quadrupole and RF gap maps all use this), and a field B_y over a
    length L changes pₓ by −qB_yL whatever the particle's momentum, so every particle must receive the node's kick −qB_yL/p₀
    unchanged. A particle with relative momentum deviation δ instead receives (1 + δ)⁻¹ of it, 1.961% less at δ = +2%. The
    TEAPOT kick, used when the linac tracker is off, is exact.

    The script tracks three on-axis protons with zero transverse momentum and relative momentum deviations −2%, 0 and +2%
    through a `DCorrectorH` with B = 0.01 T and 0.1 m effective length, once with the TEAPOT module (the control) and once
    with the linac tracker. It compares each particle's change of the horizontal coordinate with the momentum-independent
    value −0.299792qBL/p₀ that the node computes. It also checks that the linac drift advances x by L·p₀u/(β₀E), which fixes
    u = pₓ/p₀.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import math

    from orbit.core.bunch import Bunch
    from orbit.core.linac import linac_tracking
    from orbit.py_linac.lattice import DCorrectorH

    M = 0.938272013
    K0 = 0.1  # GeV
    B, LEFF = 0.01, 0.1


    def kin_for_delta(delta):
        p0 = math.sqrt(K0 * (K0 + 2 * M))
        p = p0 * (1 + delta)
        return math.sqrt(p * p + M * M) - M


    def make_bunch():
        b = Bunch()
        b.getSyncParticle().kinEnergy(K0)
        for d in (-0.02, 0.0, 0.02):
            b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, kin_for_delta(d) - K0)
        return b


    p0 = math.sqrt(K0 * (K0 + 2 * M))
    expected = (
        -B * 1.0 * LEFF * 0.299792 / p0
    )  # node formula, = Delta p_x / p0 with Delta p_x = -q B_y L (exact)
    reproduced = False
    for linac in (False, True):
        node = DCorrectorH("dch")
        node.setParam("B", B)
        node.setParam("effLength", LEFF)
        node.setLinacTracker(linac)
        b = make_bunch()
        node.trackBunch(b)
        for i, d in enumerate((-0.02, 0.0, 0.02)):
            obs = b.xp(i)
            print(
                f"{'linac' if linac else 'TEAPOT':6s} delta {d:+.2f}: kick {obs:.6e}, expected {expected:.6e} "
                f"(Delta p_x = -q B L for any momentum); ratio {obs / expected:.6f}"
            )
            if linac and d != 0.0 and abs(obs / expected - 1) > 1e-6:
                reproduced = True

    # Convention check: linac drift x_new = x + L*p0*u/(beta0*E) is the motion of a particle with p_x = p0*u.
    b = Bunch()
    b.getSyncParticle().kinEnergy(K0)
    b.addParticle(0.0, 1e-3, 0.0, 0.0, 0.0, kin_for_delta(0.02) - K0)
    linac_tracking.drift(b, 1.0)
    e_tot = kin_for_delta(0.02) + M
    beta0 = b.getSyncParticle().beta()
    print(
        f"linac drift x after 1 m: {b.x(0):.9e}, expected p0*u*L/(beta0*E) = {p0 * 1e-3 / (beta0 * e_tot):.9e}"
    )
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

