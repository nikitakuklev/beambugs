---
title: "PAR-04: A MAD-X MULTIPOLE with both KNL and KSL keeps only the array written last"
hide:
  - toc
---

# PAR-04 — A MAD-X MULTIPOLE with both KNL and KSL keeps only the array written last

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`madx_parser.py#L787-L809`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L787-L809) |

When a MAD-X MULTIPOLE gives both normal (KNL) and skew (KSL) integrated strengths, `readMADX` keeps only the array written last, with that array's normal or skew type, and drops the other with no message. MAD-X applies both: at y = 0 the quadrupole terms give a particle at x the kicks Δpx = −knl₁·x and Δpy = ksl₁·x. For `mp: multipole, knl={0, 0.1}, ksl={0, 0.05}` a particle at x = 1 mm leaves with Δpx = 0 instead of −1.000×10⁻⁴ and Δpy = 5.000×10⁻⁵; with KSL written first the skew kick is lost instead. Multipoles with a single array import correctly.

Run `python PAR-04.py`.

??? example "Reproducer: PAR-04.py"

    [Download PAR-04.py](PAR-04.py)

    ```python
    """PAR-04: A MAD-X MULTIPOLE with both KNL and KSL keeps only the array written last

    When a MAD-X MULTIPOLE gives both normal (KNL) and skew (KSL) integrated strengths, `readMADX` keeps only the array written last, with that array's normal or skew type, and drops the other with no message. MAD-X applies both: at y = 0 the quadrupole terms give a particle at x the kicks Δpx = −knl₁·x and Δpy = ksl₁·x. For `mp: multipole, knl={0, 0.1}, ksl={0, 0.05}` a particle at x = 1 mm leaves with Δpx = 0 instead of −1.000×10⁻⁴ and Δpy = 5.000×10⁻⁵; with KSL written first the skew kick is lost instead. Multipoles with a single array import correctly.

    The script reads MAD-X sequences holding one thin multipole with `readMADX`, prints the stored pole orders, strengths and skew flags, and tracks one proton at x = 1 mm. The expected kicks follow the MAD-X thin-multipole kick at y = 0, which PyORBIT's normal and skew multipole kicks also implement. The control is a multipole with KNL only.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice

    X = 1e-3
    # MAD-X thin multipole: dpx - i*dpy = -sum (knl_n + i*ksl_n) (x + i*y)^n / n!; for n = 1 at y = 0,
    # dpx = -knl1*x and dpy = +ksl1*x.
    CASES = (
        ("knl={0, 0.1}, ksl={0, 0.05}", -0.1 * X, 0.05 * X, False),
        ("ksl={0, 0.05}, knl={0, 0.1}", -0.1 * X, 0.05 * X, False),
        ("control, knl={0, 0.1}", -0.1 * X, 0.0, True),
    )
    bad = False
    for arrays, exp_px, exp_py, is_control in CASES:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seq.madx")
            with open(path, "w") as f:
                f.write(
                    f"mp: multipole, {arrays.removeprefix('control, ')};\nring: sequence, l=10;\nmp, at=2;\nendsequence;\n"
                )
            lat = TEAPOT_Lattice()
            lat.readMADX(path, "ring")
        mp = next(n for n in lat.getNodes() if n.getName() == "mp")
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(X, 0.0, 0.0, 0.0, 0.0, 0.0)
        lat.trackBunch(b)
        print(
            f"[{arrays}] stored poles {mp.getParam('poles')}, kls {mp.getParam('kls')}, skews {mp.getParam('skews')}; "
            f"px observed {b.xp(0):.6e}, expected {exp_px:.6e}; py observed {b.yp(0):.6e}, expected {exp_py:.6e}"
        )
        if not is_control:
            bad = bad or abs(b.xp(0) - exp_px) > 1e-12 or abs(b.yp(0) - exp_py) > 1e-12
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-04-madx.py"

    [Download PAR-04-madx.py](PAR-04-madx.py)

    ```python
    """PAR-04: MAD-X applies both arrays of the same multipoles

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script loads the sequences of the main reproducer into MAD-X, tracks one proton at x = 1 mm and prints the kicks. MAD-X gives Δpx = −1.000×10⁻⁴ and Δpy = 5.000×10⁻⁵ in either order of the arrays.
    """

    import os
    import tempfile

    from cpymad.madx import Madx

    with tempfile.TemporaryDirectory() as d:
        for arrays in ("knl={0, 0.1}, ksl={0, 0.05}", "ksl={0, 0.05}, knl={0, 0.1}", "knl={0, 0.1}"):
            path = os.path.join(d, "seq.madx")
            with open(path, "w") as f:
                f.write(f"mp: multipole, {arrays};\nring: sequence, l=10;\nmp, at=2;\nendsequence;\n")
            m = Madx(stdout=False)
            m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
            m.call(path)
            m.input("use, sequence=ring;")
            m.input("track, onepass=true, onetable=true; start, x=1e-3; run, turns=1; endtrack;")
            t = m.table.trackone
            print(f"MAD-X [{arrays}] px {float(t.px[-1]):.6e}, py {float(t.py[-1]):.6e}")
            m.quit()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

