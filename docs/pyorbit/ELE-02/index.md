---
title: "ELE-02: Yaw and pitch error transforms change the particle energy"
hide:
  - toc
---

# ELE-02 — Yaw and pitch error transforms change the particle energy

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Elements and apertures |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`errorbase.cc#L141-L325`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Errors/errorbase.cc#L141-L325)<br>[`errorbase.cc#L163-L169`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Errors/errorbase.cc#L163-L169)<br>[`teapotbase.cc#L120-L140`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L120-L140)<br>[`ErrorNode.py#L898-L931`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/errors/ErrorNode.py#L898-L931)<br>[`ErrorNodesAndControllersLib.py#L188-L257`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/errors/ErrorNodesAndControllersLib.py#L188-L257) |

The yaw and pitch error transforms (`XSRot`, `YSRot`) of the TEAPOT errors package, also used by the linac rotation-error nodes, rotate (p_x, 1 + δ), with δ = dE/(p₀β₀), as a momentum vector and keep its rotated second component as the new 1 + δ. A frame rotation by an angle a preserves |p| and the energy; here δ changes by sin(a)·p_x + cos(a) − 1 at the entrance, and the exit undoes this only if the element left p_x unchanged. A 1 m quadrupole with k = 2 m⁻², yawed by 1 mrad, changes the energy of a 1 GeV kinetic-energy proton entering at x = 5 mm by 10.15 keV; a yawed 1 m drift shifts a particle with p_x = 2 mrad by Δz = 3.515×10⁻⁷ m. Straight-element rolls and displacements conserve dE.

Run `python ELE-02.py`.

??? example "Reproducer: ELE-02.py"

    [Download ELE-02.py](ELE-02.py)

    ```python
    """ELE-02: Yaw and pitch error transforms change the particle energy

    The yaw and pitch error transforms (`XSRot`, `YSRot`) of the TEAPOT errors package, also used by the linac
    rotation-error nodes, rotate (p_x, 1 + δ), with δ = dE/(p₀β₀), as a momentum vector and keep its rotated second
    component as the new 1 + δ. A frame rotation by an angle a preserves |p| and the energy; here δ changes by
    sin(a)·p_x + cos(a) − 1 at the entrance, and the exit undoes this only if the element left p_x unchanged. A 1 m
    quadrupole with k = 2 m⁻², yawed by 1 mrad, changes the energy of a 1 GeV kinetic-energy proton entering at x = 5 mm by
    10.15 keV; a yawed 1 m drift shifts a particle with p_x = 2 mrad by Δz = 3.515×10⁻⁷ m. Straight-element rolls and
    displacements conserve dE.

    The script adds 1 mrad yaw and pitch errors with the public `AddErrorNode` (StraightError, subtypes XSRot and YSRot) to
    a TEAPOT quadrupole and a drift, and calls the entrance transform alone on particles with several p_x. The expected
    energy change is zero, because a frame rotation keeps |p|; for the yawed drift the expected output is the unrotated
    drift, since free flight does not depend on the frame. Controls: a particle on the yawed magnetic axis leaves on it,
    and XYRot and TransDisp errors keep dE.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.core.error_base import StraightRotationXSI  # ty: ignore[unresolved-import]
    from orbit.errors import AddErrorNode
    from orbit.teapot import DriftTEAPOT, QuadTEAPOT, TEAPOT_Lattice

    MASS, EK = 0.93827208816, 1.0  # proton, GeV
    A, L = 1.0e-3, 1.0  # error angle (rad), element length (m)


    def bunch(parts):
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(EK)
        for p in parts:
            b.addParticle(*p)
        return b


    def lattice(kind, error):
        e = DriftTEAPOT("d") if kind == "drift" else QuadTEAPOT("q")
        e.setLength(L)
        if kind == "quad":
            e.addParam("kq", 2.0)
            e.setnParts(10)
        lat = TEAPOT_Lattice(kind)
        lat.addNode(e)
        lat.initialize()
        if error is not None:
            AddErrorNode(
                lat, 0.4 * L, 0.6 * L, {"errtype": "StraightError", "sample": "Fixed", **error}
            )
        return lat


    def track(kind, error, part):
        b = bunch([part])
        lattice(kind, error).trackBunch(b)
        return b.x(0), b.xp(0), b.y(0), b.yp(0), b.z(0), b.dE(0)


    sync = bunch([]).getSyncParticle()
    P0B0 = sync.momentum() * sync.beta()  # dE = delta * p0 * beta0
    G2I = 1.0 / sync.gamma() ** 2
    reproduced = True

    # Yawed and pitched quadrupole: a static magnetic field cannot change the energy, so dE_out must be 0.
    # The source map leaves dE = -/+ a * dp * p0 beta0, with dp the change of p_x (p_y) inside the body.
    for sub, start in (
        ("XSRot", (5e-3, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ("YSRot", (0.0, 0.0, 5e-3, 0.0, 0.0, 0.0)),
    ):
        out = track("quad", {"subtype": sub, "angle": A}, start)
        dp, sign = (out[1], -1.0) if sub == "XSRot" else (out[3], 1.0)
        print(
            f"[quad k = 2 m^-2, L = 1 m, {sub} 1 mrad, start {'x' if sub == 'XSRot' else 'y'} = 5 mm] dE observed "
            f"{out[5]:.6e} GeV, expected 0; source model {sign:+.0f}*a*dp*p0*beta0 = {sign * A * dp * P0B0:.6e} GeV"
        )
        reproduced &= abs(out[5]) > 1e-6

    # Entrance transform alone: an exact rotation keeps |p| and dE; the source changes delta by sin(a) px + cos(a) - 1.
    for px in (0.0, 2e-3, -4e-3):
        b = bunch([(0.0, px, 0.0, 0.0, 0.0, 0.0)])
        StraightRotationXSI(b, A, L)
        model = (math.sin(A) * px + math.cos(A) - 1.0) * P0B0
        print(
            f"[XSI entrance, px = {px:+.0e}] dE observed {b.dE(0):+.6e} GeV, expected 0; source model {model:+.6e} GeV"
        )
        reproduced &= abs(b.dE(0)) > 1e-7

    # Yawed drift: free flight is frame independent, so z must equal the unrotated drift. The source leaves the
    # body with delta_body = sin(a) px + cos(a) - 1 and the TEAPOT drift adds L*delta_body/gamma0^2 to z.
    part = (0.0, 2e-3, 0.0, 0.0, 0.0, 0.0)
    yaw, plain = track("drift", {"subtype": "XSRot", "angle": A}, part), track("drift", None, part)
    offset = L * (math.sin(A) * part[1] + math.cos(A) - 1.0) * G2I
    print(
        f"[drift L = 1 m, XSRot 1 mrad, px = 2 mrad] z observed {yaw[4]:.6e} m, expected {plain[4]:.6e} m (unrotated drift); "
        f"difference {yaw[4] - plain[4]:.6e} m, source model L*delta_body/gamma0^2 = {offset:.6e} m"
    )
    reproduced &= abs(yaw[4] - plain[4]) > 1e-7

    # Control: the error range is centred at L/2, so a particle on the yawed axis x(s) = (s - L/2) tan(a) leaves on it.
    ax = track(
        "quad",
        {"subtype": "XSRot", "angle": A},
        (-L / 2 * math.tan(A), math.sin(A), 0.0, 0.0, 0.0, 0.0),
    )
    print(
        f"[control, quad on its yawed axis] x observed {ax[0]:.6e} m, px {ax[1]:.6e}; expected {L / 2 * math.tan(A):.6e} m, "
        f"{math.sin(A):.6e} (straight line along the axis, to O(a^3))"
    )
    reproduced &= abs(ax[0] - L / 2 * math.tan(A)) < 1e-9 and abs(ax[1] - math.sin(A)) < 1e-9

    # Controls: a roll and a transverse displacement keep the input dE exactly.
    for sub, extra in (("XYRot", {"angle": 0.3}), ("TransDisp", {"dx": 2e-3, "dy": -1e-3})):
        out = track("quad", {"subtype": sub, **extra}, (5e-3, 1e-4, -2e-3, 0.0, 0.0, 1e-4))
        print(
            f"[control, quad {sub}] dE observed {out[5]:.6e} GeV, expected 1.000000e-04 GeV (input dE)"
        )
        reproduced &= abs(out[5] - 1e-4) < 1e-15
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

