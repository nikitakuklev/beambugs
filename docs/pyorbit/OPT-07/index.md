---
title: "OPT-07: TEAPOT_MATRIX_Lattice drops the feed-down of misaligned or off-orbit nonlinear elements"
hide:
  - toc
---

# OPT-07 — `TEAPOT_MATRIX_Lattice` drops the feed-down of misaligned or off-orbit nonlinear elements

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Optics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`teapot_matrix_lattice.py#L49-L68`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot_matrix_lattice.py#L49-L68)<br>[`MatrixGenerator.cc#L58-L116`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/MatrixGenerator.cc#L58-L116)<br>[`MATRIX_Lattice.py#L53-L71`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L53-L71)<br>[`ErrorNode.py#L60-L104`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/errors/ErrorNode.py#L60-L104)<br>[`ErrorNode.py#L855-L871`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/errors/ErrorNode.py#L855-L871)<br>[`#L932-L933`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/errors/ErrorNode.py#L932-L933) |

`TEAPOT_MATRIX_Lattice` measures each node's transfer matrix from probe particles restarted on the reference axis before every node, so an orbit offset at a nonlinear element never enters its linearization. A sextupole of integrated strength K₂L displaced by (Δx, Δy) with the errors package, or on an orbit bump, loses its feed-down gradients K₂L·Δx and K₂L·Δy in the one-turn matrix, tunes and coupling, although the tracked map contains them. A thin sextupole with K₂L = 3 m⁻² displaced by (1.500, −0.900) mm has ∂x′/∂x = −4.500×10⁻³ m⁻¹ in the tracked map and 0 in the matrix lattice. In a FODO ring the matrix lattice reports the undisplaced tune 0.2698 where the tracked map has eigentunes 0.2676 and 0.2709. Linear misaligned elements are reproduced correctly.

Run `python OPT-07.py`.

??? example "Reproducer: OPT-07.py"

    [Download OPT-07.py](OPT-07.py)

    ```python
    """OPT-07: `TEAPOT_MATRIX_Lattice` drops the feed-down of misaligned or off-orbit nonlinear elements

    `TEAPOT_MATRIX_Lattice` measures each node's transfer matrix from probe particles restarted on the reference axis before every node, so an orbit offset at a nonlinear element never enters its linearization. A sextupole of integrated strength K₂L displaced by (Δx, Δy) with the errors package, or on an orbit bump, loses its feed-down gradients K₂L·Δx and K₂L·Δy in the one-turn matrix, tunes and coupling, although the tracked map contains them. A thin sextupole with K₂L = 3 m⁻² displaced by (1.500, −0.900) mm has ∂x′/∂x = −4.500×10⁻³ m⁻¹ in the tracked map and 0 in the matrix lattice. In a FODO ring the matrix lattice reports the undisplaced tune 0.2698 where the tracked map has eigentunes 0.2676 and 0.2709. Linear misaligned elements are reproduced correctly.

    The script displaces a thin sextupole between two drifts with `AddErrorNode` (TransDisp) and compares four entries of the matrix lattice's one-turn matrix with the analytic Jacobian of the displaced sextupole kick and with central differences of `trackBunch`; it repeats the comparison for a sextupole that sits on the orbit bump of an upstream kicker. It then compares the eigentunes of the matrix lattice and of the tracked Jacobian for a FODO ring with the sextupole on axis, displaced, and replaced by an explicit quadrupole and skew quadrupole of the feed-down strengths, and reports the displaced ring's closed orbit and the eigentunes about
    it. A displaced quadrupole without fringe fields is the linear control.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.errors import AddErrorNode
    from orbit.teapot import (
        DriftTEAPOT,
        KickTEAPOT,
        MultipoleTEAPOT,
        QuadTEAPOT,
        TEAPOT_Lattice,
        TEAPOT_MATRIX_Lattice,
    )

    MASS, EK = 0.93827208816, 1.0  # proton, kinetic energy in GeV
    K2L, DX, DY = 3.0, 1.5e-3, -0.9e-3  # integrated sextupole strength [m^-2], displacement [m]
    TRANSDISP = {
        "errtype": "StraightError",
        "subtype": "TransDisp",
        "sample": "Fixed",
        "dx": DX,
        "dy": DY,
    }


    def bunch(parts=()):
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(EK)
        for p in parts:
            b.addParticle(*p)
        return b


    def lattice(nodes):
        lat = TEAPOT_Lattice("lat")
        for n in nodes:
            lat.addNode(n)
        lat.initialize()
        return lat


    def thin_sext(name, k2l):
        return MultipoleTEAPOT(name, 0.0, 2, [2], [k2l], [0])


    def quad(name, length, kq, fringe=True):
        q = QuadTEAPOT(name, length, 4, kq)
        q.setUsageFringeFieldIN(fringe)
        q.setUsageFringeFieldOUT(fringe)
        return q


    def track4(lat, p):
        b = bunch([tuple(p) + (0.0, 0.0)])
        lat.trackBunch(b)
        return np.array([b.x(0), b.xp(0), b.y(0), b.yp(0)])


    def tracked_jacobian(lat, h=2e-6):
        """4x4 Jacobian of the tracked map at the reference axis: Richardson-extrapolated central
        differences with steps h and h/2 (error O(h^4))."""

        def cd(j, s):
            e = np.zeros(4)
            e[j] = s
            return (track4(lat, e) - track4(lat, -e)) / (2 * s)

        return np.column_stack([(4 * cd(j, h / 2) - cd(j, h)) / 3 for j in range(4)])


    def matrix_lattice(lat):
        mt = TEAPOT_MATRIX_Lattice(lat, bunch()).getOneTurnMatrix()
        return np.array([[mt.get(i, j) for j in range(4)] for i in range(4)])


    def eigentunes(m):
        # mode tunes from the eigenvalues exp(+-2 pi i nu); all tunes here are below 1/2
        return sorted(float(np.angle(w) / (2 * math.pi)) for w in np.linalg.eigvals(m) if w.imag > 0)


    def ring(middle):
        # FODO ring of 6.25 m with equal x and y tunes; the element under test sits after QD and D2
        return lattice(
            [
                quad("qf1", 0.25, 1.2),
                DriftTEAPOT("d1", 1.15),
                quad("qd", 0.5, -1.2),
                DriftTEAPOT("d2", 1.15),
                middle,
                DriftTEAPOT("d3", 1.15),
                DriftTEAPOT("d4", 1.6),
                quad("qf2", 0.25, 1.2),
            ]
        )


    ENTRIES = (("dpx/dx", 1, 0), ("dpx/dy", 1, 2), ("dpy/dx", 3, 0), ("dpy/dy", 3, 2))
    defects = []  # one entry per defect case: the matrix lattice misses the tracked value
    controls = []  # one entry per control: the matrix lattice equals the tracked value

    # 1. Thin sextupole displaced by (DX, DY) between two 0.3 m drifts. The TransDisp node adds (DX, DY)
    # to the coordinates, so the reference axis crosses the sextupole at body offset (xb, yb) = (DX, DY).
    # The kick dpx = -K2L (x^2 - y^2)/2, dpy = K2L x y has the Jacobian
    # d(dpx)/dx = -K2L xb, d(dpx)/dy = K2L yb, d(dpy)/dx = K2L yb, d(dpy)/dy = K2L xb, and the drifts add
    # nothing to the momentum rows' position derivatives, so these are the line's matrix entries.
    lat = lattice([DriftTEAPOT("a", 0.3), thin_sext("sx", K2L), DriftTEAPOT("b", 0.3)])
    AddErrorNode(lat, 0.29, 0.31, TRANSDISP)
    mm, jt = matrix_lattice(lat), tracked_jacobian(lat)
    analytic = {"dpx/dx": -K2L * DX, "dpx/dy": K2L * DY, "dpy/dx": K2L * DY, "dpy/dy": K2L * DX}
    print(
        f"[thin sextupole K2L = {K2L:g} m^-2 displaced by ({DX * 1e3:g}, {DY * 1e3:g}) mm with AddErrorNode]"
    )
    for key, i, j in ENTRIES:
        print(
            f"  {key}: matrix lattice {mm[i, j]:.6e}, expected {analytic[key]:.6e} 1/m (analytic feed-down); tracked {jt[i, j]:.6e}"
        )
        defects.append(abs(mm[i, j] - analytic[key]) > 1e-6 and abs(jt[i, j] - analytic[key]) < 1e-9)

    # 2. No error node: a 2 mrad kicker 0.5 m upstream puts the reference axis at xb = 0.5 m * 2 mrad
    # = 1 mm in the sextupole, so d(px)/dx = -K2L xb = -3.0e-3 1/m.
    lat = lattice(
        [
            KickTEAPOT("k", 0.0, 2, 2e-3, 0.0),
            DriftTEAPOT("a", 0.5),
            thin_sext("sx", K2L),
            DriftTEAPOT("b", 0.3),
        ]
    )
    mm, jt = matrix_lattice(lat), tracked_jacobian(lat)
    xb = 0.5 * 2e-3
    print(f"[thin sextupole on the orbit bump of a 2 mrad kicker, xb = {xb * 1e3:g} mm]")
    print(
        f"  dpx/dx: matrix lattice {mm[1, 0]:.6e}, expected {-K2L * xb:.6e} 1/m (-K2L*xb); tracked {jt[1, 0]:.6e}"
    )
    defects.append(abs(mm[1, 0] + K2L * xb) > 1e-6 and abs(jt[1, 0] + K2L * xb) < 1e-9)

    # 3. Ring tunes. Expected: the eigentunes of the tracked one-turn Jacobian. Control: an explicit thin
    # quadrupole (K1L = K2L*DX; dpx = -K1L x) and skew quadrupole (K1L = K2L*DY; dpx = K1L y) in place of
    # the displaced sextupole, which both methods must reproduce.
    pos = 0.25 + 1.15 + 0.5 + 1.15  # position of the sextupole
    on_axis = ring(thin_sext("sx", K2L))
    displaced = ring(thin_sext("sx", K2L))
    AddErrorNode(displaced, pos - 0.01, pos + 0.01, TRANSDISP)
    explicit = ring(MultipoleTEAPOT("eq", 0.0, 2, [1, 1], [K2L * DX, K2L * DY], [0, 1]))
    for tag, lat, is_control in (
        ("control, sextupole on axis", on_axis, True),
        ("sextupole displaced with AddErrorNode", displaced, False),
        ("control, explicit quadrupole + skew quadrupole of the feed-down strengths", explicit, True),
    ):
        mm, jt = matrix_lattice(lat), tracked_jacobian(lat)
        gap = np.abs(mm - jt).max()
        print(
            f"[ring, {tag}] matrix-lattice eigentunes {', '.join(f'{v:.6f}' for v in eigentunes(mm))}; "
            f"expected {', '.join(f'{v:.6f}' for v in eigentunes(jt))} (tracked Jacobian); max |M - J| {gap:.2e}"
        )
        (controls if is_control else defects).append(gap < 1e-10 if is_control else gap > 1e-3)


    def jacobian_at(lat, v, h=2e-6):
        def cd(j, s):
            e = np.zeros(4)
            e[j] = s
            return (track4(lat, v + e) - track4(lat, v - e)) / (2 * s)

        return np.column_stack([(4 * cd(j, h / 2) - cd(j, h)) / 3 for j in range(4)])


    # Information: the expected values above linearize about the trajectory that starts on the axis, which
    # the matrix lattice's affine column follows. The displaced sextupole's dipole kick moves the closed
    # orbit slightly off the axis; Newton iteration finds it, and its Jacobian gives the closed-orbit tunes.
    co = np.zeros(4)
    for _ in range(20):
        f = track4(displaced, co) - co
        if np.abs(f).max() < 1e-16:
            break
        co = co - np.linalg.solve(jacobian_at(displaced, co) - np.eye(4), f)
    print(
        f"[ring, sextupole displaced, information] closed orbit at the start x = {co[0] * 1e6:.2f} um, "
        f"y = {co[2] * 1e6:.2f} um; eigentunes about it "
        f"{', '.join(f'{v:.6f}' for v in eigentunes(jacobian_at(displaced, co)))}"
    )

    # 4. Linear control: a quadrupole (fringe fields off, so its map is linear) displaced by (DX, DY);
    # the shift nodes and the quadrupole compose exactly, so the matrix lattice equals the tracked Jacobian.
    lat = lattice([DriftTEAPOT("a", 0.3), quad("q", 0.4, 0.8, fringe=False), DriftTEAPOT("b", 0.3)])
    AddErrorNode(lat, 0.3, 0.7, TRANSDISP)
    gap_lin = np.abs(matrix_lattice(lat) - tracked_jacobian(lat)).max()
    print(
        f"[control, displaced quadrupole without fringe fields] max |M - J| = {gap_lin:.2e}, expected 0"
    )

    controls.append(gap_lin < 1e-10)
    print("DEFECT REPRODUCED" if all(defects) and all(controls) else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

