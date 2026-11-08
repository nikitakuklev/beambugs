---
title: "PAR-13: readMADX drops the skew strengths K1S, K2S and K3S"
hide:
  - toc
---

# PAR-13 — `readMADX` drops the skew strengths K1S, K2S and K3S

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot.py#L270-L278`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L270-L278)<br>[`teapot.py#L280-L298`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L280-L298) |

`TEAPOT_Lattice.readMADX` builds a QUADRUPOLE from K1 alone, a SEXTUPOLE from K2 alone and an OCTUPOLE from K3 alone, and discards their skew strengths K1S, K2S and K3S without a message. A skew quadrupole defined as `quadrupole, l=1, k1s=0.5` becomes a 1 m drift: from x = 1 mm, y = 2 mm a 3 m line ends at x = 1.000 mm instead of MAD-X's 2.555 mm, and a magnet with both K1 = 0.5 m⁻² and K1S = 0.2 m⁻² loses its coupling. A sextupole with L = 0.5 m and K2S = 6 m⁻³ gives no kick where MAD-X gives Δpx = 6.000 μrad. The same skew fields written as MULTIPOLE KSL arrays, and the normal strengths, agree with MAD-X.

Run `python PAR-13.py`.

??? example "Reproducer: PAR-13.py"

    [Download PAR-13.py](PAR-13.py)

    ```python
    """PAR-13: `readMADX` drops the skew strengths K1S, K2S and K3S

    `TEAPOT_Lattice.readMADX` builds a QUADRUPOLE from K1 alone, a SEXTUPOLE from K2 alone and an OCTUPOLE from K3 alone, and discards their skew strengths K1S, K2S and K3S without a message. A skew quadrupole defined as `quadrupole, l=1, k1s=0.5` becomes a 1 m drift: from x = 1 mm, y = 2 mm a 3 m line ends at x = 1.000 mm instead of MAD-X's 2.555 mm, and a magnet with both K1 = 0.5 m⁻² and K1S = 0.2 m⁻² loses its coupling. A sextupole with L = 0.5 m and K2S = 6 m⁻³ gives no kick where MAD-X gives Δpx = 6.000 μrad. The same skew fields written as MULTIPOLE KSL arrays, and the normal strengths, agree with MAD-X.

    The script writes MAD-X decks with each element centred in a 3 m line, reads them with `readMADX` and tracks one proton from x = 1 mm, y = 2 mm. The expected quadrupole exits come from the closed-form linear map of a normal quadrupole rolled to the angle of the complex strength K1 + i·K1S; the expected sextupole and octupole exits come from the MAD-X thin multipole kick at the element centre. Controls are the normal-strength quadrupole and sextupole and a thin MULTIPOLE with KSL.
    """

    import cmath
    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice

    MASS = 0.93827208816  # proton mass, GeV
    START = np.array([1e-3, 0.0, 2e-3, 0.0])  # x, px, y, py at the line entrance


    def drift(length):
        m = np.eye(4)
        m[0, 1] = m[2, 3] = length
        return m


    def quad_line(k1, k1s, length):
        """Exit of the 3 m line holding a centred quadrupole of complex strength K1 + i*K1S.

        A normal quadrupole of strength |K| rolled by theta about s has the complex strength
        |K| exp(-2i theta) (the MAD-X rule: TILT = pi/4 turns a positive normal quadrupole into a
        negative skew one), so K1 + i*K1S = |K| exp(i alpha) is that quadrupole rolled by -alpha/2.
        """
        kk = abs(complex(k1, k1s))
        theta = -cmath.phase(complex(k1, k1s)) / 2.0
        w = math.sqrt(kk)
        m = np.zeros((4, 4))
        m[0:2, 0:2] = [
            [math.cos(w * length), math.sin(w * length) / w],
            [-w * math.sin(w * length), math.cos(w * length)],
        ]
        m[2:4, 2:4] = [
            [math.cosh(w * length), math.sinh(w * length) / w],
            [w * math.sinh(w * length), math.cosh(w * length)],
        ]
        c, s = math.cos(theta), math.sin(theta)
        r = np.array(
            [[c, 0, -s, 0], [0, c, 0, -s], [s, 0, c, 0], [0, s, 0, c]]
        )  # rotation by theta, x toward y
        side = drift((3.0 - length) / 2)
        return side @ r @ m @ r.T @ side @ START


    def thin_line(n, knl, ksl):
        """Exit of the 3 m line with a thin multipole kick of order n at its centre (MAD-X convention):
        dpx - i dpy = -(knl + i ksl) (x + i y)^n / n!."""
        x, px, y, py = drift(1.5) @ START
        s = -complex(knl, ksl) * complex(x, y) ** n / math.factorial(n)
        px, py = px + s.real, py - s.imag
        return np.array([x + 1.5 * px, px, y + 1.5 * py, py])


    CASES = (
        # (label, element, expected exit, control?)
        ("quadrupole L=1 K1S=0.5", "quadrupole, l=1.0, k1s=0.5", quad_line(0.0, 0.5, 1.0), False),
        (
            "quadrupole L=1 K1=0.5 K1S=0.2",
            "quadrupole, l=1.0, k1=0.5, k1s=0.2",
            quad_line(0.5, 0.2, 1.0),
            False,
        ),
        ("sextupole L=0.5 K2S=6", "sextupole, l=0.5, k2s=6.0", thin_line(2, 0.0, 6.0 * 0.5), False),
        ("octupole L=0.5 K3S=60", "octupole, l=0.5, k3s=60.0", thin_line(3, 0.0, 60.0 * 0.5), False),
        ("control, quadrupole L=1 K1=0.5", "quadrupole, l=1.0, k1=0.5", quad_line(0.5, 0.0, 1.0), True),
        (
            "control, sextupole L=0.5 K2=6",
            "sextupole, l=0.5, k2=6.0",
            thin_line(2, 6.0 * 0.5, 0.0),
            True,
        ),
        (
            "control, multipole KSL={0, 0, 3}",
            "multipole, ksl={0, 0, 3.0}",
            thin_line(2, 0.0, 3.0),
            True,
        ),
    )


    def fmt(v):
        return "(" + ", ".join(f"{u:+.6e}" for u in v) + ")"


    bad, controls_ok = False, True
    for label, element, expected, is_control in CASES:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "line.madx")
            with open(path, "w") as f:
                f.write(
                    f"e1: {element};\nseq: sequence, l=3, refer=centre;\ne1, at=1.5;\nendsequence;\n"
                )
            lat = TEAPOT_Lattice()
            lat.readMADX(path, "seq")
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(START[0], START[1], START[2], START[3], 0.0, 0.0)
        lat.trackBunch(b)
        obs = np.array([b.x(0), b.xp(0), b.y(0), b.yp(0)])
        dev = float(np.max(np.abs(obs - expected)))
        print(
            f"[{label}] exit (x, px, y, py) observed {fmt(obs)}, expected {fmt(expected)}; max |diff| {dev:.3e}"
        )
        if is_control:
            controls_ok &= dev < 1e-8
        else:
            bad |= dev > 2e-8
    print("DEFECT REPRODUCED" if bad and controls_ok else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-13-madx.py"

    [Download PAR-13-madx.py](PAR-13-madx.py)

    ```python
    """PAR-13: MAD-X applies K1S, K2S and K3S

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script places each element of the main reproducer in the same 3 m line and computes the exit orbit from x = 1 mm, y = 2 mm with TWISS, which matches the main reproducer's expected values. It also shows that MAD-X ignores K1S on an SBEND, the case the reader leaves out correctly.
    """

    from cpymad.madx import Madx

    E = 1.0 + 0.93827208816  # total energy of a 1 GeV proton
    CASES = (
        ("quadrupole L=1 K1S=0.5", "quadrupole, l=1.0, k1s=0.5"),
        ("quadrupole L=1 K1=0.5 K1S=0.2", "quadrupole, l=1.0, k1=0.5, k1s=0.2"),
        ("sextupole L=0.5 K2S=6", "sextupole, l=0.5, k2s=6.0"),
        ("octupole L=0.5 K3S=60", "octupole, l=0.5, k3s=60.0"),
        ("quadrupole L=1 K1=0.5", "quadrupole, l=1.0, k1=0.5"),
        ("sbend L=1 ANGLE=0.1 K1S=0.4", "sbend, l=1.0, angle=0.1, k1s=0.4"),
        ("sbend L=1 ANGLE=0.1", "sbend, l=1.0, angle=0.1"),
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

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

