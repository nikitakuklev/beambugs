---
title: "OPT-02: getChromaticitiesXY differentiates about the reference axis instead of the off-momentum closed orbit"
hide:
  - toc
---

# OPT-02 — `getChromaticitiesXY` differentiates about the reference axis instead of the off-momentum closed orbit

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Optics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`MatrixGenerator.cc#L118-L170`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/MatrixGenerator.cc#L118-L170)<br>[`teapot_matrix_lattice.py#L209-L225`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot_matrix_lattice.py#L209-L225) |

`TEAPOT_MATRIX_Lattice.getChromaticitiesXY` takes the energy derivative of the one-turn matrix from probe particles that start on the reference axis with an energy offset, not on the off-momentum closed orbit. With sextupoles and nonzero dispersion at the lattice start, the sextupole part of the chromaticity ξ = dν/dδ (ν the tune, δ the relative momentum deviation) then depends on where the lattice starts. In an eight-cell FODO ring with bends and thin sextupoles, and with the ring tune restored in the method's denominator, it gives ξx, ξy = 6.562, 6.641 when the ring starts at its first element and 9.578, 5.655 when it starts at its first bend. The closed-orbit tune derivative is 7.592, 5.978 at both starts.

Run `python OPT-02.py`.

??? example "Reproducer: OPT-02.py"

    [Download OPT-02.py](OPT-02.py)

    ```python
    """OPT-02: `getChromaticitiesXY` differentiates about the reference axis instead of the off-momentum closed orbit

    `TEAPOT_MATRIX_Lattice.getChromaticitiesXY` takes the energy derivative of the one-turn matrix from
    probe particles that start on the reference axis with an energy offset, not on the off-momentum
    closed orbit. With sextupoles and nonzero dispersion at the lattice start, the sextupole part of the
    chromaticity ξ = dν/dδ (ν the tune, δ the relative momentum deviation) then depends on where the
    lattice starts. In an eight-cell FODO ring with bends and thin sextupoles, and with the ring tune
    restored in the method's denominator, it gives ξx, ξy = 6.562, 6.641 when the ring starts at its
    first element and 9.578, 5.655 when it starts at its first bend. The closed-orbit tune derivative is
    7.592, 5.978 at both starts.

    The script builds the ring with and without sextupoles and starts it at its first element and at
    its first bend. It removes the separate first-slice tune error from `getChromaticitiesXY` by
    rescaling with sin(2πν₁)/sin(2πν), then compares with central differences in δ of the tune of the
    tracked one-turn Jacobian about the off-momentum closed orbit, found by Newton iteration, and
    about the zero orbit. The ring without sextupoles is the control.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import math

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.teapot import (
        BendTEAPOT,
        DriftTEAPOT,
        MultipoleTEAPOT,
        QuadTEAPOT,
        TEAPOT_Lattice,
        TEAPOT_MATRIX_Lattice,
    )


    def make_ring(k2f, k2d, start):
        """Eight FODO cells with two 0.1 rad bends and thin sextupoles (k2*l) per cell, rotated so that
        the lattice starts at element index `start` of the first cell."""
        nodes = []
        for c in range(8):
            nodes += [
                QuadTEAPOT(f"qf{c}", 0.25, 4, 0.8),
                MultipoleTEAPOT(f"sf{c}", 0.0, 2, [2], [k2f], [0]),
                DriftTEAPOT(f"a{c}", 0.5),
                BendTEAPOT(f"b{c}", 1.5, 4, theta=0.1),
                DriftTEAPOT(f"c{c}", 0.5),
                MultipoleTEAPOT(f"sd{c}", 0.0, 2, [2], [k2d], [0]),
                QuadTEAPOT(f"qd{c}", 0.5, 4, -0.8),
                DriftTEAPOT(f"e{c}", 0.5),
                BendTEAPOT(f"g{c}", 1.5, 4, theta=0.1),
                DriftTEAPOT(f"h{c}", 0.5),
                QuadTEAPOT(f"qe{c}", 0.25, 4, 0.8),
            ]
        lat = TEAPOT_Lattice("ring")
        for n in nodes[start:] + nodes[:start]:
            lat.addNode(n)
        lat.initialize()
        return lat


    def proton_bunch():
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)  # GeV
        return b


    def track(lat, ref, v, dE):
        b = Bunch()
        ref.copyEmptyBunchTo(b)
        b.addParticle(v[0], v[1], v[2], v[3], 0.0, dE)
        lat.trackBunch(b)
        return np.array([b.x(0), b.xp(0), b.y(0), b.yp(0)])


    def jacobian(lat, ref, v, dE, h=1e-7):
        J = np.zeros((4, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = h
            J[:, j] = (track(lat, ref, v + e, dE) - track(lat, ref, v - e, dE)) / (2 * h)
        return J


    def closed_orbit(lat, ref, dE):
        # Newton iteration for the transverse fixed point at fixed dE (no RF: dE is conserved)
        v = np.zeros(4)
        for _ in range(30):
            f = track(lat, ref, v, dE) - v
            if np.max(np.abs(f)) < 1e-15:
                break
            v = v - np.linalg.solve(jacobian(lat, ref, v, dE) - np.eye(4), f)
        return v


    def fractional_tunes(J):
        # cos(mu) from the trace; the sign of M01 (= beta sin mu) selects mu in [0, 2 pi)
        out = []
        for i in (0, 2):
            c = (J[i, i] + J[i + 1, i + 1]) / 2
            s = np.sign(J[i, i + 1]) * math.sqrt(1 - c * c)
            out.append((math.atan2(s, c) / (2 * math.pi)) % 1.0)
        return out


    bad = False
    results = {}
    for k2f, k2d in ((0.0, 0.0), (2.0, -3.0)):
        for start, where in ((0, "first element"), (3, "first bend")):
            lat = make_ring(k2f, k2d, start)
            ref = proton_bunch()
            ml = TEAPOT_MATRIX_Lattice(lat, ref)
            D0 = ml.getRingParametersDict()["dispersion x [m]"]
            xi = ml.getChromaticitiesXY()
            nu = ml.getTunes()
            nu_1 = (ml.getRingTwissDataX()[0][1][1], ml.getRingTwissDataY()[0][1][1])
            # Remove the separate first-slice tune error: the formula needs sin(2 pi nu_ring)
            obs = [
                xi[k] * math.sin(2 * math.pi * nu_1[k]) / math.sin(2 * math.pi * nu[k]) for k in (0, 1)
            ]
            sp = ref.getSyncParticle()
            pb = sp.momentum() ** 2 / (sp.kinEnergy() + sp.mass())  # dE = (p^2/E) * delta
            dd = 1e-5
            co, zo = [], []
            for s in (1.0, -1.0):
                dE = s * dd * pb
                co.append(fractional_tunes(jacobian(lat, ref, closed_orbit(lat, ref, dE), dE)))
                zo.append(fractional_tunes(jacobian(lat, ref, np.zeros(4), dE)))
            # xi = d(nu)/d(delta) about the off-momentum closed orbit (the definition, independent of
            # the start point) and about the zero orbit (what the method's probe particles follow)
            exp = [(co[0][k] - co[1][k]) / (2 * dd) for k in (0, 1)]
            zero = [(zo[0][k] - zo[1][k]) / (2 * dd) for k in (0, 1)]
            tag = "sextupoles" if k2f else "control, no sextupoles"
            print(f"[{tag}, start at {where}] D at start {D0:.6f} m")
            for k, plane in enumerate("xy"):
                print(
                    f"  xi_{plane}: observed {obs[k]:.4f}, expected {exp[k]:.4f} "
                    f"(closed-orbit tune derivative); zero-orbit tune derivative {zero[k]:.4f}"
                )
                if k2f and abs(obs[k] - exp[k]) > 0.05 * abs(exp[k]):
                    bad = True
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

