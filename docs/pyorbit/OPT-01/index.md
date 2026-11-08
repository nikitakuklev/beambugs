---
title: "OPT-01: getChromaticitiesXY divides by the sine of the first-slice phase advance instead of the ring tune"
hide:
  - toc
---

# OPT-01 — `getChromaticitiesXY` divides by the sine of the first-slice phase advance instead of the ring tune

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Optics |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot_matrix_lattice.py#L203-L239`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot_matrix_lattice.py#L203-L239)<br>[`MATRIX_Lattice.py#L283-L288`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L283-L288)<br>[`teapot_matrix_lattice.py#L241-L251`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot_matrix_lattice.py#L241-L251) |

`TEAPOT_MATRIX_Lattice.getChromaticitiesXY` converts the energy derivative of the one-turn matrix trace into the chromaticity ξ = dν/dδ, with ν the betatron tune and δ the relative momentum deviation, by dividing by 2·sin(2πν). It takes ν from the second entry of the integrated phase-advance table, the phase advance ν₁ over the first slice of the lattice, instead of the ring tune, so the result is scaled by sin(2πν)/sin(2πν₁) and grows as the first slice gets shorter. For a six-cell quadrupole FODO ring with tunes 1.4753 it returns ξx = −18.62 instead of −1.830, the derivative of the tracked one-turn tune. For the SNS ring lattice shipped with the code it returns −97.36 instead of −7.548.

Run `python OPT-01.py`.

??? example "Reproducer: OPT-01.py"

    [Download OPT-01.py](OPT-01.py)

    ```python
    """OPT-01: `getChromaticitiesXY` divides by the sine of the first-slice phase advance instead of the ring tune

    `TEAPOT_MATRIX_Lattice.getChromaticitiesXY` converts the energy derivative of the one-turn matrix
    trace into the chromaticity ξ = dν/dδ, with ν the betatron tune and δ the relative momentum
    deviation, by dividing by 2·sin(2πν). It takes ν from the second entry of the integrated
    phase-advance table, the phase advance ν₁ over the first slice of the lattice, instead of the ring
    tune, so the result is scaled by sin(2πν)/sin(2πν₁) and grows as the first slice gets shorter. For a
    six-cell quadrupole FODO ring with tunes 1.4753 it returns ξx = −18.62 instead of −1.830, the
    derivative of the tracked one-turn tune. For the SNS ring lattice shipped with the code it returns
    −97.36 instead of −7.548.

    The script builds a six-cell thick-quadrupole FODO ring without bends, so the reference axis is the
    closed orbit at every energy, and compares `getChromaticitiesXY` with a central difference in δ of
    the tune of the tracked one-turn matrix. It repeats the case with quadrupoles cut into eight slices,
    which shortens the first slice. As a control it evaluates the method's own formula with the ring
    tune in the denominator, which reproduces the finite difference.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import math

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.teapot import (
        DriftTEAPOT,
        QuadTEAPOT,
        TEAPOT_Lattice,
        TEAPOT_MATRIX_Lattice,
    )


    def proton_bunch():
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)  # GeV
        return b


    def fodo_ring(nparts):
        lat = TEAPOT_Lattice("fodo")
        for c in range(6):
            lat.addNode(QuadTEAPOT(f"qf{c}", 0.25, nparts, 1.2))
            lat.addNode(DriftTEAPOT(f"d{c}a", 2.0))
            lat.addNode(QuadTEAPOT(f"qd{c}", 0.5, nparts, -1.2))
            lat.addNode(DriftTEAPOT(f"d{c}b", 2.0))
            lat.addNode(QuadTEAPOT(f"qe{c}", 0.25, nparts, 1.2))
        lat.initialize()
        return lat


    def one_turn(lat, ref, dE, h=1e-7):
        """Transverse 4x4 one-turn Jacobian about x = x' = y = y' = 0 at energy offset dE [GeV]."""
        M = np.zeros((4, 4))
        for j in range(4):
            out = []
            for s in (1.0, -1.0):
                b = Bunch()
                ref.copyEmptyBunchTo(b)
                c = [0.0] * 6
                c[j] = s * h
                c[5] = dE
                b.addParticle(*c)
                lat.trackBunch(b)
                out.append(np.array([b.x(0), b.xp(0), b.y(0), b.yp(0)]))
            M[:, j] = (out[0] - out[1]) / (2 * h)
        return M


    bad = False
    for nparts in (2, 8):
        lat = fodo_ring(nparts)
        ref = proton_bunch()
        ml = TEAPOT_MATRIX_Lattice(lat, ref)
        xi = ml.getChromaticitiesXY()
        nu_ring = ml.getTunes()
        # The method reads tune_x[1][-1], the phase table's second (s, phase) pair: the phase advance
        # accumulated over the first positive-length slice of the lattice.
        nu_1 = (ml.getRingTwissDataX()[0][1][1], ml.getRingTwissDataY()[0][1][1])
        sp = ref.getSyncParticle()
        p, E = sp.momentum(), sp.kinEnergy() + sp.mass()
        # Chromaticity xi = d(nu)/d(delta). With no bends the zero orbit is closed at every energy, so
        # the one-turn Jacobian at energy offset dE gives the off-momentum tune. dE = (p^2/E)*delta is
        # the linear relation the method itself uses. Both fractional tunes are below 1/2 (0.4753), so
        # acos((M00 + M11)/2)/2pi is the fractional tune without a branch choice.
        dd = 1e-5
        nu = []
        for s in (1.0, -1.0):
            M = one_turn(lat, ref, s * dd * p * p / E)
            nu.append([math.acos((M[i, i] + M[i + 1, i + 1]) / 2) / (2 * math.pi) for i in (0, 2)])
        fd = [(nu[0][k] - nu[1][k]) / (2 * dd) for k in (0, 1)]
        print(
            f"[nparts={nparts}] ring tunes (getTunes) {nu_ring[0]:.6f} {nu_ring[1]:.6f}; "
            f"first-slice phase advance {nu_1[0]:.6f} {nu_1[1]:.6f}"
        )
        for k, plane in enumerate("xy"):
            print(
                f"  xi_{plane}: observed {xi[k]:.6f}, expected {fd[k]:.6f} "
                "(central difference in delta of the tracked one-turn tune)"
            )
            # Control: the method divides by 2 sin(2 pi nu_1); the chromaticity formula
            # xi = -(p^2/E) d(M00 + M11)/d(dE) / (4 pi sin 2 pi nu) needs the ring tune nu, which
            # enters only through sin(2 pi nu). Rescaling restores it.
            ctl = xi[k] * math.sin(2 * math.pi * nu_1[k]) / math.sin(2 * math.pi * nu_ring[k])
            print(
                f"  control xi_{plane} with the ring tune in the denominator: observed {ctl:.6f}, "
                f"expected {fd[k]:.6f}"
            )
            if abs(xi[k] - fd[k]) > 1e-3 * abs(fd[k]):
                bad = True
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

??? example "Reproducer: OPT-01-sns.py"

    [Download OPT-01-sns.py](OPT-01-sns.py)

    ```python
    """OPT-01: the error on the SNS ring lattice shipped with the code

    Setup: set `PYORBIT_SOURCE` to a PyORBIT3 source checkout; the script reads the SNS ring lattice
    `tests/py/orbit/parsers/inputs/sns_ring.lat` from it.

    The script imports the SNS ring at 1.3 GeV, calls `getChromaticitiesXY` and compares it with a
    central difference in δ of the tune of the tracked one-turn Jacobian about the off-momentum closed
    orbit, which it finds by Newton iteration. As a control it evaluates the method's formula with the
    ring tune in the denominator.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import math

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice, TEAPOT_MATRIX_Lattice

    src = os.environ.get("PYORBIT_SOURCE")
    if not src:
        raise SystemExit("set PYORBIT_SOURCE to a PyORBIT3 source checkout")
    lat = TEAPOT_Lattice()
    lat.readMADX(os.path.join(src, "tests/py/orbit/parsers/inputs/sns_ring.lat"), "rnginjsol")
    lat.initialize()
    ref = Bunch()
    ref.mass(0.938272)
    ref.charge(1)
    ref.getSyncParticle().kinEnergy(1.3)  # GeV


    def track(v, dE):
        b = Bunch()
        ref.copyEmptyBunchTo(b)
        b.addParticle(v[0], v[1], v[2], v[3], 0.0, dE)
        lat.trackBunch(b)
        return np.array([b.x(0), b.xp(0), b.y(0), b.yp(0)])


    def jacobian(v, dE, h=1e-7):
        J = np.zeros((4, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = h
            J[:, j] = (track(v + e, dE) - track(v - e, dE)) / (2 * h)
        return J


    def closed_orbit(dE):
        # Newton iteration for the transverse fixed point at fixed dE (no RF: dE is conserved)
        v = np.zeros(4)
        for _ in range(30):
            f = track(v, dE) - v
            if np.max(np.abs(f)) < 1e-15:
                break
            v = v - np.linalg.solve(jacobian(v, dE) - np.eye(4), f)
        return v


    def fractional_tunes(J):
        # cos(mu) from the trace; the sign of M01 (= beta sin mu) selects mu in [0, 2 pi)
        out = []
        for i in (0, 2):
            c = (J[i, i] + J[i + 1, i + 1]) / 2
            s = np.sign(J[i, i + 1]) * math.sqrt(1 - c * c)
            out.append((math.atan2(s, c) / (2 * math.pi)) % 1.0)
        return out


    ml = TEAPOT_MATRIX_Lattice(lat, ref)
    xi = ml.getChromaticitiesXY()
    nu_ring = ml.getTunes()
    nu_1 = (ml.getRingTwissDataX()[0][1][1], ml.getRingTwissDataY()[0][1][1])
    sp = ref.getSyncParticle()
    pb = sp.momentum() ** 2 / (sp.kinEnergy() + sp.mass())  # dE = (p^2/E) * delta, as in the method
    dd = 1e-5
    nu = [fractional_tunes(jacobian(closed_orbit(s * dd * pb), s * dd * pb)) for s in (1.0, -1.0)]
    fd = [(nu[0][k] - nu[1][k]) / (2 * dd) for k in (0, 1)]  # xi = d(nu)/d(delta) on the closed orbit
    print(
        f"ring tunes (getTunes) {nu_ring[0]:.6f} {nu_ring[1]:.6f}; "
        f"first-slice phase advance {nu_1[0]:.6f} {nu_1[1]:.6f}"
    )
    bad = False
    for k, plane in enumerate("xy"):
        print(
            f"xi_{plane}: observed {xi[k]:.4f}, expected {fd[k]:.4f} "
            "(central difference in delta of the closed-orbit one-turn tune)"
        )
        ctl = xi[k] * math.sin(2 * math.pi * nu_1[k]) / math.sin(2 * math.pi * nu_ring[k])
        print(f"control xi_{plane} with the ring tune in the denominator: {ctl:.4f}")
        if abs(xi[k] - fd[k]) > 0.01 * abs(fd[k]):
            bad = True
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small class="bb-scope">source reading</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source reading |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

