---
title: "ELE-05: Linac aperture nodes ignore their documented centre offsets c and d"
hide:
  - toc
---

# ELE-05 — Linac aperture nodes ignore their documented centre offsets c and d

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Elements and apertures |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`LinacApertureNodes.py#L23-L61`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacApertureNodes.py#L23-L61)<br>[`LinacApertureNodes.py#L100-L124`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacApertureNodes.py#L100-L124)<br>[`wrap_PrimitiveApertureShape.cc#L66-L93`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Apertures/wrap_PrimitiveApertureShape.cc#L66-L93)<br>[`TeapotApertureNode.py#L50-L61`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/aperture/TeapotApertureNode.py#L50-L61) |

`LinacApertureNode` and its circle, ellipse and rectangle subclasses document c and d as the x and y offsets of the aperture centre, store them, and never pass them to the aperture shape, which stays centred on the axis. A circular linac aperture of radius 10 mm with c = 20 mm and d = −5 mm removes a particle sitting at its centre (20, −5) mm and keeps one on the axis, 20.62 mm from the centre; the TEAPOT `CircleApertureNode` with the same arguments keeps the first and removes the second. Ellipse and rectangle nodes behave the same way; with c = d = 0 the linac and TEAPOT nodes agree.

Run `python ELE-05.py`.

??? example "Reproducer: ELE-05.py"

    [Download ELE-05.py](ELE-05.py)

    ```python
    """ELE-05: Linac aperture nodes ignore their documented centre offsets c and d

    `LinacApertureNode` and its circle, ellipse and rectangle subclasses document c and d as the x and y offsets of the
    aperture centre, store them, and never pass them to the aperture shape, which stays centred on the axis. A circular
    linac aperture of radius 10 mm with c = 20 mm and d = −5 mm removes a particle sitting at its centre (20, −5) mm and
    keeps one on the axis, 20.62 mm from the centre; the TEAPOT `CircleApertureNode` with the same arguments keeps the first
    and removes the second. Ellipse and rectangle nodes behave the same way; with c = d = 0 the linac and TEAPOT nodes agree.

    The script builds the three linac aperture nodes with offsets (20, −5) mm and tracks one particle at the stated centre
    and one on the axis. It then passes a grid of 936 points through each shape with offsets (−12, 4) mm and compares the
    survivors with an analytic membership test centred at (c, d). Controls: the TEAPOT nodes with the same arguments, the
    linac node after its shape centre is set by hand, and the centred (c = d = 0) linac and TEAPOT nodes.
    """

    import itertools
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.aperture import CircleApertureNode, EllipseApertureNode, RectangleApertureNode
    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.py_linac.lattice import (
        CircleLinacApertureNode,
        EllipseLinacApertureNode,
        RectangleLinacApertureNode,
    )

    MASS, EK = 0.93827208816, 0.4


    def survivors(node, pts):
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(EK)
        for x, y in pts:
            b.addParticle(x, 0.0, y, 0.0, 0.0, 0.0)
        lost = Bunch()
        b.copyEmptyBunchTo(lost)
        node.track({"bunch": b, "lostbunch": lost})
        return {(round(b.x(i), 12), round(b.y(i), 12)) for i in range(b.getSize())}


    def inside(kind, x, y, a, b):
        # Membership of a shape of half sizes (a, b) centred at the origin, after subtracting the centre (c, d)
        if kind == "circle":
            return x * x + y * y <= a * a
        if kind == "ellipse":
            return (x / a) ** 2 + (y / b) ** 2 <= 1.0
        return abs(x) <= a and abs(y) <= b


    # (kind, half sizes, linac node factory, TEAPOT node factory); the second positional argument is the position.
    SHAPES = (
        (
            "circle",
            (0.010, 0.010),
            lambda a, b, c, d: CircleLinacApertureNode(a, 0.0, c, d),
            lambda a, b, c, d: CircleApertureNode(a, 0.0, c, d),
        ),
        (
            "ellipse",
            (0.010, 0.008),
            lambda a, b, c, d: EllipseLinacApertureNode(a, b, 0.0, c, d),
            lambda a, b, c, d: EllipseApertureNode(a, b, 0.0, c, d),
        ),
        (
            "rectangle",
            (0.010, 0.008),
            lambda a, b, c, d: RectangleLinacApertureNode(a, b, 0.0, c, d),
            lambda a, b, c, d: RectangleApertureNode(a, b, 0.0, c, d),
        ),
    )
    # Off-boundary grid, about 1.3 mm pitch, x in [-24, 24] mm and y in [-17, 17] mm
    GRID = [
        (0.00031 + i * 0.00137, -0.00023 + j * 0.00131)
        for i, j in itertools.product(range(-18, 18), range(-13, 13))
    ]
    GRID_SIZES = {"circle": (0.007, 0.007), "ellipse": (0.010, 0.006), "rectangle": (0.010, 0.006)}

    defects, controls = 0, True
    for kind, (a, b), linac, teapot in SHAPES:
        c, d = 0.020, -0.005
        pts = [(c, d), (0.0, 0.0)]
        exp = {(c, d)}  # the centre point is inside, the axis point is 20.62 mm from the centre
        obs, ref = survivors(linac(a, b, c, d), pts), survivors(teapot(a, b, c, d), pts)
        print(
            f"[linac {kind}, c = 20 mm, d = -5 mm] survivors observed {sorted(obs)}; expected {sorted(exp)}; TEAPOT node {sorted(ref)}"
        )
        defects += obs != exp
        controls &= ref == exp

        ga, gb = GRID_SIZES[kind]
        c, d = -0.012, 0.004
        exp = {(round(x, 12), round(y, 12)) for x, y in GRID if inside(kind, x - c, y - d, ga, gb)}
        centred = {(round(x, 12), round(y, 12)) for x, y in GRID if inside(kind, x, y, ga, gb)}
        obs = survivors(linac(ga, gb, c, d), GRID)
        fixed = linac(ga, gb, c, d)
        shape = fixed.getBaseAperture().getApertureShape()
        shape.centerX(c)
        shape.centerY(d)
        by_hand, ref = survivors(fixed, GRID), survivors(teapot(ga, gb, c, d), GRID)
        print(
            f"[linac {kind}, c = -12 mm, d = 4 mm, {len(GRID)} grid points] survivors observed {len(obs)} (set equals the "
            f"centred aperture: {obs == centred}), expected {len(exp)} (analytic, centred at (c, d)); with the shape centre set "
            f"by hand {len(by_hand)} (equals expected: {by_hand == exp}); TEAPOT node {len(ref)} (equals expected: {ref == exp})"
        )
        defects += obs != exp
        controls &= by_hand == exp and ref == exp

        cpts = [(0.0, 0.0), (0.009, 0.0), (0.0, 0.0075), (0.012, 0.0), (0.0, -0.0085)]
        lc, tc = survivors(linac(a, b, 0.0, 0.0), cpts), survivors(teapot(a, b, 0.0, 0.0), cpts)
        print(f"    [control, c = d = 0] linac survivors {sorted(lc)}; TEAPOT survivors {sorted(tc)}")
        controls &= lc == tc
    print("DEFECT REPRODUCED" if defects == 6 and controls else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

