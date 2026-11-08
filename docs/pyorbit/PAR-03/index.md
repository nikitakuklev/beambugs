---
title: "PAR-03: HKICKER and VKICKER lose their KICK strength on MAD and MAD-X import"
hide:
  - toc
---

# PAR-03 — HKICKER and VKICKER lose their KICK strength on MAD and MAD-X import

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot.py#L310-L333`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L310-L333)<br>[`madx_parser.py#L766-L767`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L766-L767) |

In MAD-X and MAD-8 the strength of an HKICKER or VKICKER is its KICK attribute, the change of px or py; only the two-plane KICKER uses HKICK and VKICK. The TEAPOT element factory reads only `hkick` and `vkick`, so through `readMADX` and `readMAD` every single-plane corrector defined with KICK is built with zero kick and no message. For `hk: hkicker, kick=0.001` and `vk: vkicker, kick=0.002` an on-axis particle leaves with px = py = 0 instead of 1.000×10⁻³ and 2.000×10⁻³. Correctors written with `hkick=` or `vkick=` import correctly.

Run `python PAR-03.py`.

??? example "Reproducer: PAR-03.py"

    [Download PAR-03.py](PAR-03.py)

    ```python
    """PAR-03: HKICKER and VKICKER lose their KICK strength on MAD and MAD-X import

    In MAD-X and MAD-8 the strength of an HKICKER or VKICKER is its KICK attribute, the change of px or py; only the two-plane KICKER uses HKICK and VKICK. The TEAPOT element factory reads only `hkick` and `vkick`, so through `readMADX` and `readMAD` every single-plane corrector defined with KICK is built with zero kick and no message. For `hk: hkicker, kick=0.001` and `vk: vkicker, kick=0.002` an on-axis particle leaves with px = py = 0 instead of 1.000×10⁻³ and 2.000×10⁻³. Correctors written with `hkick=` or `vkick=` import correctly.

    The script writes a MAD-X sequence and an equivalent MAD-8 line holding the two correctors to a temporary directory, reads each with its reader and tracks one on-axis proton. The expected px and py are the KICK values, the momentum change these correctors define. The control repeats the MAD-X sequence with the correctors written as `hkick=` and `vkick=`.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice

    MADX_TAIL = "ring: sequence, l=10;\nhk, at=4;\nvk, at=5;\nendsequence;\n"
    CASES = (
        (
            "MAD-X, kick=",
            "readMADX",
            "seq.madx",
            "hk: hkicker, l=0, kick=0.001;\nvk: vkicker, l=0, kick=0.002;\n" + MADX_TAIL,
            False,
        ),
        (
            "MAD-8, kick=",
            "readMAD",
            "line.mad",
            "hk: hkicker, l=0.0, kick=0.001\nvk: vkicker, l=0.0, kick=0.002\nd: drift, l=1.0\nring: line=(d, hk, vk, d)\n",
            False,
        ),
        (
            "control, MAD-X, hkick= and vkick=",
            "readMADX",
            "seq.madx",
            "hk: hkicker, l=0, hkick=0.001;\nvk: vkicker, l=0, vkick=0.002;\n" + MADX_TAIL,
            True,
        ),
    )
    bad = False
    for tag, reader, name, text, is_control in CASES:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name)
            with open(path, "w") as f:
                f.write(text)
            lat = TEAPOT_Lattice()
            getattr(lat, reader)(path, "ring")
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        lat.trackBunch(b)
        # HKICKER: dpx = KICK = 1e-3; VKICKER: dpy = KICK = 2e-3; drifts keep px and py.
        print(
            f"[{tag}] px observed {b.xp(0):.6e}, expected 1.000000e-03; py observed {b.yp(0):.6e}, expected 2.000000e-03"
        )
        if not is_control:
            bad = bad or abs(b.xp(0) - 1e-3) > 1e-12 or abs(b.yp(0) - 2e-3) > 1e-12
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-03-madx.py"

    [Download PAR-03-madx.py](PAR-03-madx.py)

    ```python
    """PAR-03: MAD-X applies KICK on an HKICKER and a VKICKER and saves correctors with KICK

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script loads the MAD-X sequence of the main reproducer, tracks one on-axis proton and prints px and py, then saves the sequence and prints the corrector definitions MAD-X writes. MAD-X gives px = 1.000×10⁻³ and py = 2.000×10⁻³ and writes the correctors with `kick`.
    """

    import os
    import tempfile

    from cpymad.madx import Madx

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.madx")
        saved = os.path.join(d, "saved.seq")
        with open(path, "w") as f:
            f.write(
                "hk: hkicker, l=0, kick=0.001;\nvk: vkicker, l=0, kick=0.002;\nring: sequence, l=10;\nhk, at=4;\nvk, at=5;\nendsequence;\n"
            )
        m = Madx(stdout=False)
        m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
        m.call(path)
        m.input("use, sequence=ring;")
        m.input("track, onepass=true, onetable=true; start; run, turns=1; endtrack;")
        t = m.table.trackone
        print(f"MAD-X px {float(t.px[-1]):.6e} (KICK 1e-3), py {float(t.py[-1]):.6e} (KICK 2e-3)")
        m.input(f"save, sequence=ring, file='{saved}';")
        m.quit()
        with open(saved) as f:
            print("MAD-X save:", [line.strip() for line in f if "kicker" in line])

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

