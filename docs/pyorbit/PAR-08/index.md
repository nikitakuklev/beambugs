---
title: "PAR-08: The TEAPOT element factory stops on bare TILT flags, zero bend angles and RCOLLIMATOR"
hide:
  - toc
---

# PAR-08 — The TEAPOT element factory stops on bare TILT flags, zero bend angles and RCOLLIMATOR

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot.py#L244-L256`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L244-L256)<br>[`teapot.py#L276-L298`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L276-L298)<br>[`teapot.py#L352-L360`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L352-L360)<br>[`teapot.py#L376-L380`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L376-L380)<br>[`teapot.py#L1193`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1193)<br>[`mad_parser.py#L34`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/mad_parser.py#L34) |

Three valid MAD element definitions stop the TEAPOT lattice readers. A bend with ANGLE = 0 or without ANGLE, such as a field-free or pure-gradient sector magnet, raises ZeroDivisionError in `readMAD` and `readMADX` alike, because the bend's radius is computed as L/θ and the RBEND length conversion divides by sin(θ/2). A TILT without a value, which MAD-8 defines as the element's standard skew angle (π/4 for a quadrupole), raises AttributeError in `readMAD` for bends, quadrupoles, sextupoles and octupoles. RCOLLIMATOR, which the MAD-8 parser accepts, matches no factory branch because the factory tests for the misspelling "rcolimator", and the process exits with status 1.

Run `python PAR-08.py`.

??? example "Reproducer: PAR-08.py"

    [Download PAR-08.py](PAR-08.py)

    ```python
    """PAR-08: The TEAPOT element factory stops on bare TILT flags, zero bend angles and RCOLLIMATOR

    Three valid MAD element definitions stop the TEAPOT lattice readers. A bend with ANGLE = 0 or without ANGLE, such as a field-free or pure-gradient sector magnet, raises ZeroDivisionError in `readMAD` and `readMADX` alike, because the bend's radius is computed as L/θ and the RBEND length conversion divides by sin(θ/2). A TILT without a value, which MAD-8 defines as the element's standard skew angle (π/4 for a quadrupole), raises AttributeError in `readMAD` for bends, quadrupoles, sextupoles and octupoles. RCOLLIMATOR, which the MAD-8 parser accepts, matches no factory branch because the factory tests for the misspelling "rcolimator", and the process exits with status 1.

    The script writes one MAD-8 line or MAD-X sequence per case, a 1 m drift and the 1 m element under test, and reads it in a child interpreter, because the factory ends the process with `sys.exit` for an unknown type. It prints the child's exit status and its last error line. The expected result is a 2 m lattice that reads without error; the control is a quadrupole with TILT = 0.3.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import subprocess
    import sys
    import tempfile

    CHILD = (
        "import sys\nfrom orbit.teapot import TEAPOT_Lattice\nlat = TEAPOT_Lattice()\n"
        "getattr(lat, sys.argv[2])(sys.argv[1], 'ring')\nprint('read ok', lat.getLength())\n"
    )
    MAD8 = "\nd: drift, l=1.0\nring: line=(d, q)\n"
    MADX = "\nring: sequence, l=2;\nq, at=1.5;\nendsequence;\n"
    CASES = (  # (label, reader, element definition, control)
        ("quadrupole with bare TILT", "readMAD", "q: quadrupole, l=1.0, k1=0.5, tilt", False),
        ("sextupole with bare TILT", "readMAD", "q: sextupole, l=1.0, k2=0.5, tilt", False),
        ("octupole with bare TILT", "readMAD", "q: octupole, l=1.0, k3=0.5, tilt", False),
        ("sbend with bare TILT", "readMAD", "q: sbend, l=1.0, angle=0.1, tilt", False),
        ("sbend with angle 0 and k1", "readMAD", "q: sbend, l=1.0, angle=0.0, k1=0.3", False),
        ("sbend without angle", "readMAD", "q: sbend, l=1.0, k1=0.3", False),
        ("rbend with angle 0", "readMAD", "q: rbend, l=1.0, angle=0.0", False),
        ("rcollimator", "readMAD", "q: rcollimator, l=1.0", False),
        ("sbend with angle 0 and k1", "readMADX", "q: sbend, l=1.0, angle=0.0, k1=0.3;", False),
        ("rbend with angle 0", "readMADX", "q: rbend, l=1.0, angle=0.0;", False),
        (
            "control, quadrupole with TILT=0.3",
            "readMAD",
            "q: quadrupole, l=1.0, k1=0.5, tilt=0.3",
            True,
        ),
    )
    bad = False
    with tempfile.TemporaryDirectory() as d:
        for tag, reader, elem, is_control in CASES:
            path = os.path.join(d, "lattice.mad" if reader == "readMAD" else "seq.madx")
            with open(path, "w") as f:
                f.write(elem + (MAD8 if reader == "readMAD" else MADX))
            r = subprocess.run(
                [sys.executable, "-I", "-c", CHILD, path, reader],
                capture_output=True,
                text=True,
                check=False,
            )
            lines = [line.strip() for line in (r.stdout + r.stderr).splitlines() if line.strip()]
            refusal = [line for line in lines if "Can not create" in line]
            last = refusal[0] if refusal else (lines[-1] if lines else "")
            print(
                f"[{reader}, {tag}] observed exit {r.returncode}, '{last}'; expected exit 0, 'read ok 2.0'"
            )
            if not is_control:
                bad = bad or r.returncode != 0
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-08-madx.py"

    [Download PAR-08-madx.py](PAR-08-madx.py)

    ```python
    """PAR-08: MAD-X reads the zero-angle bends and a bare TILT

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script loads 2 m MAD-X sequences holding a 1 m sector bend with ANGLE = 0 and k1, a sector bend without ANGLE, a rectangular bend with ANGLE = 0 and a quadrupole with a bare TILT, and prints the one-pass R₁₁ that MAD-X computes and the TILT it stores. MAD-X reads all four; it stores a bare TILT as 0.
    """

    import os
    import tempfile

    from cpymad.madx import Madx

    ELEMENTS = (
        "q: sbend, l=1.0, angle=0.0, k1=0.3;",
        "q: sbend, l=1.0, k1=0.3;",
        "q: rbend, l=1.0, angle=0.0;",
        "q: quadrupole, l=1.0, k1=0.5, tilt;",
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.madx")
        for elem in ELEMENTS:
            with open(path, "w") as f:
                f.write(elem + "\nring: sequence, l=2;\nq, at=1.5;\nendsequence;\n")
            m = Madx(stdout=False)
            m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
            m.call(path)
            m.input("use, sequence=ring;")
            m.input("twiss, betx=1, bety=1, rmatrix;")
            print(
                f"MAD-X [{elem}] read ok, R11 {m.table.twiss.re11[-1]:.6f}, tilt {m.elements['q'].tilt}"
            )
            m.quit()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

