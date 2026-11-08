---
title: "PAR-12: MAD and MAD-X import reverses the sense of TILT"
hide:
  - toc
---

# PAR-12 — MAD and MAD-X import reverses the sense of TILT

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot.py#L210-L217`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L210-L217)<br>[`teapot.py#L383-L385`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L383-L385)<br>[`teapot.py#L467-L473`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L467-L473)<br>[`teapot.py#L1443-L1450`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1443-L1450)<br>[`teapotbase.cc#L80-L101`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L80-L101) |

`readMADX` and `readMAD` pass an element's TILT unchanged to the TEAPOT tilt nodes, whose rotation has the opposite sense to MAD's positive roll (x toward y), so every tilted element is built rolled by −TILT. A kicker with HKICK = 1 mrad and TILT = 0.4 kicks y′ by −0.3894 mrad where MAD-X gives +0.3894 mrad. A quadrupole with K1 = 0.5 m⁻² and TILT = 0.3 couples the planes with the wrong sign: a 3 m line from x = 1 mm, y = 2 mm ends at x = 1.280 mm instead of −0.4162 mm. A sector bend with TILT = π/2 disperses to −y instead of +y. Untilted elements, and tilted ones whose angle is negated after import, agree with MAD-X to 2×10⁻⁹ m.

Run `python PAR-12.py`.

??? example "Reproducer: PAR-12.py"

    [Download PAR-12.py](PAR-12.py)

    ```python
    """PAR-12: MAD and MAD-X import reverses the sense of TILT

    `readMADX` and `readMAD` pass an element's TILT unchanged to the TEAPOT tilt nodes, whose rotation has the opposite sense to MAD's positive roll (x toward y), so every tilted element is built rolled by −TILT. A kicker with HKICK = 1 mrad and TILT = 0.4 kicks y′ by −0.3894 mrad where MAD-X gives +0.3894 mrad. A quadrupole with K1 = 0.5 m⁻² and TILT = 0.3 couples the planes with the wrong sign: a 3 m line from x = 1 mm, y = 2 mm ends at x = 1.280 mm instead of −0.4162 mm. A sector bend with TILT = π/2 disperses to −y instead of +y. Untilted elements, and tilted ones whose angle is negated after import, agree with MAD-X to 2×10⁻⁹ m.

    The script writes MAD-X and MAD-8 decks with a tilted kicker, quadrupole, thin multipole and sector bend centred in a 3 m line, reads them and tracks one proton. The expected exit coordinates are those of the element rolled by +TILT, R(TILT)·M·R(−TILT), with M the untilted map in closed form; for the bend, the expected exit point is the untilted bend's exit point rotated by π/2. Controls are the untilted kicker and each tilted lattice with its tilt angle negated after import.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice

    MASS = 0.93827208816  # proton mass, GeV
    START = (1e-3, 0.0, 2e-3, 0.0)  # x, px, y, py at the line entrance


    def rot(a):
        """Rotation of (x, px, y, py) by a, x toward y: x' = cos a x - sin a y (likewise for px, py)."""
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]])


    def drift(length):
        m = np.eye(4)
        m[0, 1] = m[2, 3] = length
        return m


    def quad(k, length):
        """Linear map of a normal quadrupole of strength k > 0 (focusing in x)."""
        w = math.sqrt(k)
        m = np.zeros((4, 4))
        m[0:2, 0:2] = [
            [math.cos(w * length), math.sin(w * length) / w],
            [-w * math.sin(w * length), math.cos(w * length)],
        ]
        m[2:4, 2:4] = [
            [math.cosh(w * length), math.sinh(w * length) / w],
            [w * math.sinh(w * length), math.cosh(w * length)],
        ]
        return m


    def thin_quad(kl):
        """Thin normal quadrupole: dpx = -kl x, dpy = +kl y."""
        m = np.eye(4)
        m[1, 0], m[3, 2] = -kl, kl
        return m


    def rolled(m, tilt):
        # An element rolled by +tilt about s (MAD-X and MAD-8 positive TILT, x toward y) maps lab
        # coordinates to its own frame with R(-tilt) and back with R(+tilt).
        return rot(tilt) @ m @ rot(-tilt)


    v0 = np.array(START)
    # Kicker: in its own frame the kick is (HKICK, 0); in the lab it is R(tilt)(HKICK, 0) = HKICK (cos, sin).
    hk, tk = 1e-3, 0.4
    kick = drift(1.5) @ v0 + np.array(
        [1.5 * hk * math.cos(tk), hk * math.cos(tk), 1.5 * hk * math.sin(tk), hk * math.sin(tk)]
    )
    CASES = (
        # (label, reader, deck, expected exit (x, px, y, py) or None for the bend, control?)
        (
            "readMADX, kicker HKICK=1e-3 TILT=0.4",
            "madx",
            f"kicker, l=0, hkick=1e-3, tilt={tk}",
            kick,
            False,
        ),
        (
            "readMAD, kicker HKICK=1e-3 TILT=0.4",
            "mad",
            f"KICKER, L=0.0, HKICK=1e-3, TILT={tk}",
            kick,
            False,
        ),
        (
            "readMADX, quadrupole K1=0.5 TILT=0.3",
            "madx",
            "quadrupole, l=1.0, k1=0.5, tilt=0.3",
            drift(1.0) @ rolled(quad(0.5, 1.0), 0.3) @ drift(1.0) @ v0,
            False,
        ),
        (
            "readMAD, quadrupole K1=0.5 TILT=0.3",
            "mad",
            "QUADRUPOLE, L=1.0, K1=0.5, TILT=0.3",
            drift(1.0) @ rolled(quad(0.5, 1.0), 0.3) @ drift(1.0) @ v0,
            False,
        ),
        (
            "readMADX, multipole KNL={0, 0.3} TILT=0.7",
            "madx",
            "multipole, knl={0, 0.3}, tilt=0.7",
            drift(1.5) @ rolled(thin_quad(0.3), 0.7) @ drift(1.5) @ v0,
            False,
        ),
        (
            "control, readMADX, kicker HKICK=1e-3 TILT=0",
            "madx",
            "kicker, l=0, hkick=1e-3, tilt=0.0",
            drift(1.5) @ v0 + np.array([1.5 * hk, hk, 0.0, 0.0]),
            True,
        ),
    )


    def read(reader, element, length):
        with tempfile.TemporaryDirectory() as d:
            if reader == "madx":
                path = os.path.join(d, "line.madx")
                text = f"e1: {element};\nseq: sequence, l=3, refer=centre;\ne1, at=1.5;\nendsequence;\n"
            else:
                path = os.path.join(d, "line.mad")
                text = f"D: DRIFT, L={(3.0 - length) / 2}\nE1: {element}\nSEQ: LINE=(D, E1, D)\n"
            with open(path, "w") as f:
                f.write(text)
            lat = TEAPOT_Lattice()
            if reader == "madx":
                lat.readMADX(path, "seq")
            else:
                lat.readMAD(path, "SEQ")
        return lat


    def track(lat, v, dE=0.0):
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(v[0], v[1], v[2], v[3], 0.0, dE)
        lat.trackBunch(b)
        return np.array([b.x(0), b.xp(0), b.y(0), b.yp(0)])


    def negate_tilt(lat):
        for node in lat.getNodes():
            if node.getName().lower() == "e1":
                node.setTiltAngle(-node.getTiltAngle())


    def fmt(v):
        return "(" + ", ".join(f"{u:+.6e}" for u in v) + ")"


    bad, controls_ok = False, True
    for label, reader, element, expected, is_control in CASES:
        length = 1.0 if "L=1.0" in element or "l=1.0" in element else 0.0
        lat = read(reader, element, length)
        obs = track(lat, v0)
        dev = float(np.max(np.abs(obs - expected)))
        print(
            f"[{label}] exit (x, px, y, py) observed {fmt(obs)}, expected {fmt(expected)} (element rolled by +TILT); max |diff| {dev:.3e}"
        )
        if is_control:
            controls_ok &= dev < 1e-8
            continue
        bad |= dev > 1e-5
        negate_tilt(lat)
        dev_neg = float(np.max(np.abs(track(lat, v0) - expected)))
        print(f"    [control, tilt negated after import] max |diff| {dev_neg:.3e}")
        controls_ok &= dev_neg < 1e-8

    # Sector bend rolled by +pi/2: a particle entering on axis with dE = 1 MeV leaves the untilted
    # bend at (x_u, y_u = 0); rolling the element by +pi/2 turns its x axis into the lab y axis, so
    # the rolled bend must leave it at (x, y) = R(pi/2)(x_u, y_u) = (-y_u, x_u): dispersion toward +y.
    untilted = track(read("madx", "sbend, l=1.0, angle=0.1", 1.0), (0, 0, 0, 0), dE=1e-3)
    expected = rot(math.pi / 2) @ untilted
    lat = read("madx", f"sbend, l=1.0, angle=0.1, tilt={math.pi / 2!r}", 1.0)
    obs = track(lat, (0, 0, 0, 0), dE=1e-3)
    dev = float(np.max(np.abs(obs - expected)))
    print(
        f"[readMADX, sbend ANGLE=0.1 TILT=pi/2, dE = 1 MeV] exit y observed {obs[2]:+.6e} m, expected {expected[2]:+.6e} m "
        f"(untilted bend's x rotated by pi/2); max |diff| {dev:.3e}"
    )
    bad |= dev > 1e-6
    negate_tilt(lat)
    dev_neg = float(np.max(np.abs(track(lat, (0, 0, 0, 0), dE=1e-3) - expected)))
    print(f"    [control, tilt negated after import] max |diff| {dev_neg:.3e}")
    controls_ok &= dev_neg < 1e-12
    print("DEFECT REPRODUCED" if bad and controls_ok else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-12-madx.py"

    [Download PAR-12-madx.py](PAR-12-madx.py)

    ```python
    """PAR-12: MAD-X tracks the tilted elements of the main reproducer as rolled by +TILT

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script places the tilted kicker, quadrupole and thin multipole of the main reproducer in the same 3 m line and computes the exit orbit from x = 1 mm, y = 2 mm with TWISS, which matches the main reproducer's expected values for an element rolled by +TILT. It also tracks a particle with dE = 1 MeV through the 1 m sector bend with TILT = π/2, which leaves at positive y.
    """

    import math

    from cpymad.madx import Madx

    E = 1.0 + 0.93827208816  # total energy of a 1 GeV proton
    CASES = (
        ("kicker HKICK=1e-3 TILT=0.4", "kicker, l=0, hkick=1e-3, tilt=0.4"),
        ("quadrupole K1=0.5 TILT=0.3", "quadrupole, l=1.0, k1=0.5, tilt=0.3"),
        ("multipole KNL={0, 0.3} TILT=0.7", "multipole, knl={0, 0.3}, tilt=0.7"),
    )
    for label, element in CASES:
        m = Madx(stdout=False)
        m.input(
            f"beam, particle=proton, energy={E!r};\ne1: {element};\nseq: sequence, l=3, refer=centre;\ne1, at=1.5;\nendsequence;\nuse, sequence=seq;"
        )
        tw = m.twiss(betx=10, bety=10, x=1e-3, px=0, y=2e-3, py=0)
        exit_ = ", ".join(f"{float(tw[k][-1]):+.6e}" for k in ("x", "px", "y", "py"))
        print(f"MAD-X [{label}] exit (x, px, y, py) ({exit_})")
        m.quit()

    m = Madx(stdout=False)
    m.input(
        f"beam, particle=proton, energy={E!r};\ne1: sbend, l=1.0, angle=0.1, tilt={math.pi / 2!r};\n"
        "seq: sequence, l=3, refer=centre;\ne1, at=1.5;\nendsequence;\nuse, sequence=seq;"
    )
    p0 = math.sqrt(E * E - 0.93827208816**2)
    m.input(f"track, onepass=true, onetable=true; start, pt={1e-3 / p0!r}; run, turns=1; endtrack;")
    t = m.table.trackone
    print(
        f"MAD-X [sbend ANGLE=0.1 TILT=pi/2, dE = 1 MeV] exit x {float(t.x[-1]):+.6e} m, y {float(t.y[-1]):+.6e} m"
    )
    m.quit()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

