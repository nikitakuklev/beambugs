---
title: "TRK-13: The edge wedges of a combined-function bend scale their multipole kick by the bend length in metres"
hide:
  - toc
---

# TRK-13 — The edge wedges of a combined-function bend scale their multipole kick by the bend length in metres

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`teapotbase.cc#L1484-L1503`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L1484-L1503)<br>[`teapot.py#L1099-L1110`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1099-L1110)<br>[`teapot.py#L1136-L1147`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1136-L1147)<br>[`teapot.py#L1091-L1096`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1091-L1096)<br>[`LinacAccNodes.py#L548-L576`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L548-L576)<br>[`LinacAccNodes.py#L594-L609`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L594-L609) |

In a TEAPOT or linac combined-function bend of length L with a nonzero face angle, the thin wedge between the sector body and the inclined face is kicked with the magnet's integrated multipole strength KₙL times the wedge length s, where the strength per length Kₙ gives Kₙ·s, so the wedge kick is L/(1 m) times its correct value. A 2 m bend with K₁ = 0.4 m⁻² and 0.1 rad faces must have the map of its two halves cut at a perpendicular plane, yet a ray at x = 10 mm leaves it 6.007 µm in x and 1.577 µrad in x′ away from them; passing KₙL/L makes them agree to roundoff. The error is second order in x, so linear optics are unaffected.

Run `python TRK-13.py`.

??? example "Reproducer: TRK-13.py"

    [Download TRK-13.py](TRK-13.py)

    ```python
    """TRK-13: The edge wedges of a combined-function bend scale their multipole kick by the bend length in metres

    In a TEAPOT or linac combined-function bend of length L with a nonzero face angle, the thin wedge between the sector body and the inclined face is kicked with the magnet's integrated multipole strength KₙL times the wedge length s, where the strength per length Kₙ gives Kₙ·s, so the wedge kick is L/(1 m) times its correct value. A 2 m bend with K₁ = 0.4 m⁻² and 0.1 rad faces must have the map of its two halves cut at a perpendicular plane, yet a ray at x = 10 mm leaves it 6.007 µm in x and 1.577 µrad in x′ away from them; passing KₙL/L makes them agree to roundoff. The error is second order in x, so linear optics are unaffected.

    The script calls the wedge kernel `wedgebendCF` directly for a ray crossing an entrance wedge of a pure quadrupole field and compares its kick with the exact thick-quadrupole result over the wedge path, for magnet lengths of 0.5, 1 and 2 m. It then tracks rays through a 2 m `BendTEAPOT` and a 2 m linac `Bend` and through the same magnets built as two halves joined at a perpendicular face with its fringe disabled. The controls are the same comparisons with K₁ = 0 and, for TEAPOT, with copies of the shipped fringe functions that pass KₙL/L to the wedge.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.py_linac.lattice import LinacAccLattice, Sequence
    from orbit.py_linac.lattice.LinacAccNodes import Bend
    from orbit.teapot import teapot
    from orbit.teapot_base import TPB

    MASS = 0.93827208816
    L, THETA, K1, E = 2.0, 0.2, 0.4, 0.1  # m, rad, m^-2, rad
    STEP_PARTS = 512  # body parts per metre; integration error below 1e-11 here
    RAYS = [(1e-2, 0.0, 0.0, 0.0, 0.0, 0.0), (-1e-2, 1e-4, 3e-3, -1e-4, 0.0, 0.0)]


    def make_bunch(rays):
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(1.0)
        for r in rays:
            b.addParticle(*r)
        return b


    # ---------------------------------------------------------------- the wedge kernel alone
    # A ray at x0 with x' = 0 crosses the entrance wedge on the signed path s = -x0 tan(e); with rho -> infinity only
    # the quadrupole term acts, and a uniform quadrupole of per-length strength K1 over s gives x' = -sqrt(K1) sin(sqrt(K1) s) x0.
    defects, controls_ok = 0, True
    x0 = 5e-3
    s = -x0 * math.tan(E)
    exp_xp = -math.sqrt(K1) * math.sin(math.sqrt(K1) * s) * x0
    for length in (0.5, 1.0, 2.0):
        b = make_bunch([(x0, 0.0, 0.0, 0.0, 0.0, 0.0)])
        TPB.wedgebendCF(
            b, E, 0, 1e30, 1, [1], [K1 * length], [0], 4000, 1
        )  # the callers pass kl = K1*L
        ratio = b.xp(0) / exp_xp
        print(
            f"[wedgebendCF, kl = K1*L, L = {length:.1f} m] x' observed {b.xp(0):.6e}, expected {exp_xp:.6e} "
            f"(thick quadrupole K1 over the wedge path s = {s:.6e} m); ratio {ratio:.6f}, L/(1 m) = {length:.1f}"
        )
        if length == 1.0:
            controls_ok &= abs(ratio - 1) < 1e-6
        else:
            defects += abs(ratio - 1) > 1e-3


    # ---------------------------------------------------------------- split invariance, TEAPOT
    def corrected_fringes(bend):
        """Copies of BendTEAPOT's fringeIN/fringeOUT with K_nL/L passed to wedgebendCF."""

        def fin(node, p):
            e = node.getParam("ea1")
            rho = node.getParam("rho")
            poles, kls, skews = node.getParam("poles"), node.getParam("kls"), node.getParam("skews")
            length, b, n = p["parentNode"].getLength(), p["bunch"], p["parentNode"].getnParts()
            if e != 0.0:
                TPB.wedgedrift(b, e, 0)
                if node.getUsage():
                    TPB.wedgerotate(b, e, 0)
                    TPB.bendfringeIN(b, rho)
                    for i in range(len(poles)):
                        TPB.multpfringeIN(b, poles[i], kls[i] / length, skews[i], 1)
                    TPB.wedgerotate(b, e, 1)
                TPB.wedgebendCF(
                    b, e, 0, rho, len(poles), poles, [k / length for k in kls], skews, n - 1, 1
                )
            elif node.getUsage():
                TPB.bendfringeIN(b, rho)
                for i in range(len(poles)):
                    TPB.multpfringeIN(b, poles[i], kls[i] / length, skews[i], 1)

        def fout(node, p):
            e = node.getParam("ea2")
            rho = node.getParam("rho")
            poles, kls, skews = node.getParam("poles"), node.getParam("kls"), node.getParam("skews")
            length, b, n = p["parentNode"].getLength(), p["bunch"], p["parentNode"].getnParts()
            if e != 0.0:
                TPB.wedgebendCF(
                    b, e, 1, rho, len(poles), poles, [k / length for k in kls], skews, n - 1, 1
                )
                if node.getUsage():
                    TPB.wedgerotate(b, -e, 0)
                    TPB.bendfringeOUT(b, rho)
                    for i in range(len(poles)):
                        TPB.multpfringeOUT(b, poles[i], kls[i] / length, skews[i], 1)
                    TPB.wedgerotate(b, -e, 1)
                TPB.wedgedrift(b, e, 1)
            elif node.getUsage():
                TPB.bendfringeOUT(b, rho)
                for i in range(len(poles)):
                    TPB.multpfringeOUT(b, poles[i], kls[i] / length, skews[i], 1)

        bend.setFringeFieldFunctionIN(fin)
        bend.setFringeFieldFunctionOUT(fout)


    def teapot_bend(length, theta, k1, e1, e2, fr_in, fr_out, fixed):
        poles, kls, skews = ([1], [k1 * length], [0]) if k1 else ([], [], [])
        b = teapot.BendTEAPOT(
            "b",
            length=length,
            nparts=round(STEP_PARTS * length) + 1,
            theta=theta,
            ea1=e1,
            ea2=e2,
            poles=poles,
            kls=kls,
            skews=skews,
        )
        b.setUsageFringeFieldIN(fr_in)
        b.setUsageFringeFieldOUT(fr_out)
        if fixed:
            corrected_fringes(b)
        return b


    def track_teapot(nodes):
        lat = teapot.TEAPOT_Lattice()
        for n in nodes:
            lat.addNode(n)
        lat.initialize()
        b = make_bunch(RAYS)
        lat.trackBunch(b)
        return [(b.x(i), b.xp(i), b.y(i), b.yp(i)) for i in range(b.getSize())]


    def linac_bend(length, theta, k1, e1, e2, fr_in, fr_out):
        b = Bend("b")
        b.setLength(length)
        b.setnParts(round(STEP_PARTS * length) + 1)
        # the linac Bend passes -kls*charge*L as the integrated strength, so kls = -K1 is a focusing K1
        b.setParam("poles", [1] if k1 else [])
        b.setParam("kls", [-k1] if k1 else [])
        b.setParam("skews", [0] if k1 else [])
        b.setParam("ea1", e1)
        b.setParam("ea2", e2)
        b.setParam("theta", theta)
        b.setUsageFringeFieldIN(fr_in)
        b.setUsageFringeFieldOUT(fr_out)
        return b


    def track_linac(nodes):
        lat = LinacAccLattice("l")
        seq = Sequence("s")
        for n in nodes:
            seq.addNode(n)
            lat.addNode(n)
        lat.initialize()
        b = make_bunch(RAYS)
        lat.trackBunch(b)
        return [(b.x(i), b.xp(i), b.y(i), b.yp(i)) for i in range(b.getSize())]


    def split(code, k1, fixed=False):
        # whole magnet against two halves joined at a perpendicular face whose fringe is off: the same magnet, same step
        if code == "TEAPOT":
            whole = track_teapot([teapot_bend(L, THETA, k1, E, E, True, True, fixed)])
            halves = track_teapot(
                [
                    teapot_bend(L / 2, THETA / 2, k1, E, 0.0, True, False, fixed),
                    teapot_bend(L / 2, THETA / 2, k1, 0.0, E, False, True, fixed),
                ]
            )
        else:
            whole = track_linac([linac_bend(L, THETA, k1, E, E, True, True)])
            halves = track_linac(
                [
                    linac_bend(L / 2, THETA / 2, k1, E, 0.0, True, False),
                    linac_bend(L / 2, THETA / 2, k1, 0.0, E, False, True),
                ]
            )
        d0 = [a - c for a, c in zip(whole[0], halves[0])]
        worst = max(abs(a - c) for rw, rh in zip(whole, halves) for a, c in zip(rw, rh))
        return d0, worst


    for code, k1, fixed, label in (
        ("TEAPOT", K1, False, "BendTEAPOT"),
        ("linac", K1, False, "linac Bend"),
        ("TEAPOT", 0.0, False, "BendTEAPOT, K1 = 0 (control)"),
        ("linac", 0.0, False, "linac Bend, K1 = 0 (control)"),
        ("TEAPOT", K1, True, "BendTEAPOT, wedge given K1L/L (control)"),
    ):
        d0, worst = split(code, k1, fixed)
        print(
            f"[{label}, L = {L} m, K1 = {k1} m^-2, e1 = e2 = {E}] whole minus halves, ray x = 10 mm: dx {d0[0]:+.4e} m, "
            f"dx' {d0[1]:+.4e}; max over rays {worst:.4e}; expected 0 (same magnet, same step)"
        )
        if "control" in label:
            controls_ok &= worst < 1e-12
        else:
            defects += worst > 1e-7

    print("DEFECT REPRODUCED" if defects == 4 and controls_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

