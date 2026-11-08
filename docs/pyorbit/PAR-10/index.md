---
title: "PAR-10: readMADX multiplies numbers written with a decimal point and an exponent by a power of ten"
hide:
  - toc
---

# PAR-10 — `readMADX` multiplies numbers written with a decimal point and an exponent by a power of ten

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P1 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`madx_parser.py#L703-L712`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L703-L712)<br>[`madx_parser.py#L277-L278`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L277-L278)<br>[`madx_parser.py#L331`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L331)<br>[`madx_parser.py#L215-L216`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L215-L216)<br>[`madx_parser.py#L436-L446`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L436-L446) |

`TEAPOT_Lattice.readMADX` misreads every number in an element attribute that has both a decimal point and an exponent: it takes `1.5e` in `1.5e-3` for a variable name, deletes its '.', and evaluates `15e-3`. The value is multiplied by 10 to the power of the number of digits after the point, with no message: `k1=1.5e-3` gives k₁ = 0.015 m⁻² instead of 0.0015 m⁻², and the MAD-X SAVE output `hkick:= -1.25e-05` gives a kick of −1.250×10⁻³ instead of −1.250×10⁻⁵. In a 16-cell ring saved by MAD-X with eight orbit correctors, the one-pass orbit reaches x = 24.93 mm instead of 1.156 mm. Integer mantissas, fixed-point decimals, variable definitions, AT positions and KNL/KSL arrays are read correctly.

Run `python PAR-10.py`.

??? example "Reproducer: PAR-10.py"

    [Download PAR-10.py](PAR-10.py)

    ```python
    """PAR-10: `readMADX` multiplies numbers written with a decimal point and an exponent by a power of ten

    `TEAPOT_Lattice.readMADX` misreads every number in an element attribute that has both a decimal point and an exponent: it takes `1.5e` in `1.5e-3` for a variable name, deletes its '.', and evaluates `15e-3`. The value is multiplied by 10 to the power of the number of digits after the point, with no message: `k1=1.5e-3` gives k₁ = 0.015 m⁻² instead of 0.0015 m⁻², and the MAD-X SAVE output `hkick:= -1.25e-05` gives a kick of −1.250×10⁻³ instead of −1.250×10⁻⁵. In a 16-cell ring saved by MAD-X with eight orbit correctors, the one-pass orbit reaches x = 24.93 mm instead of 1.156 mm. Integer mantissas, fixed-point decimals, variable definitions, AT positions and KNL/KSL arrays are read correctly.

    The script reads small MAD-X sequences whose element attributes are written as exponent literals and compares the imported strengths with the literal values. It then writes a 16-cell FODO ring with eight kicker correctors in the form MAD-X SAVE produces, reads it, and tracks one proton from the origin through it once. The expected values are the literals themselves and, for the ring, the same deck with every literal written in fixed-point notation, which is the same set of numbers. Controls are literals without a decimal point or exponent, a global variable defined with an exponent literal and a KNL array.
    """

    import contextlib
    import decimal
    import io
    import os
    import re

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice

    MASS = 0.938272088  # proton mass, GeV
    TAIL = "seq: sequence, l=10, refer=centre;\n{name}, at=2;\nendsequence;\n"

    # (label, element definition (preceded by any variable definitions), element name, node parameter,
    #  expected value = the literal as MAD-X evaluates it, control?)
    CASES = (
        ("quadrupole k1=1.5e-3", "q: quadrupole, l=1, k1=1.5e-3;\n", "q", "kq", 1.5e-3, False),
        ("quadrupole k1=0.25e-3", "q: quadrupole, l=1, k1=0.25e-3;\n", "q", "kq", 0.25e-3, False),
        (
            "quadrupole k1:=1.2345678901e-05",
            "q: quadrupole, l=1, k1:=1.2345678901e-05;\n",
            "q",
            "kq",
            1.2345678901e-05,
            False,
        ),
        (
            "SAVE form, kicker hkick:= -1.25e-05",
            "c: kicker, l:= 0;\nc1: c,hkick:= -1.25e-05;\n",
            "c1",
            "kx",
            -1.25e-05,
            False,
        ),
        (
            "SAVE form, kicker vkick:= 3.7e-05",
            "c: kicker, l:= 0;\nc1: c,vkick:= 3.7e-05;\n",
            "c1",
            "ky",
            3.7e-05,
            False,
        ),
        ("sbend angle=1.5e-3", "b: sbend, l=1, angle=1.5e-3;\n", "b", "theta", 1.5e-3, False),
        ("control, quadrupole k1=1e-3", "q: quadrupole, l=1, k1=1e-3;\n", "q", "kq", 1e-3, True),
        ("control, quadrupole k1=0.0015", "q: quadrupole, l=1, k1=0.0015;\n", "q", "kq", 0.0015, True),
        (
            "control, kq = 1.5e-3; k1:=kq",
            "kq = 1.5e-3;\nq: quadrupole, l=1, k1:=kq;\n",
            "q",
            "kq",
            1.5e-3,
            True,
        ),
        (
            "control, multipole knl={0, 1.5e-3}",
            "q: multipole, knl={0, 1.5e-3};\n",
            "q",
            "kls",
            1.5e-3,
            True,
        ),
    )


    def read(text):
        """Read a MAD-X deck from a scratch file; return the lattice and what the reader printed."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "deck.madx")
            with open(path, "w") as f:
                f.write(text)
            lat = TEAPOT_Lattice()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                lat.readMADX(path, "seq" if "seq:" in text else "ring")
        return lat, buf.getvalue()


    bad, controls_ok = False, True
    for label, definition, name, param, expected, is_control in CASES:
        lat, out = read(definition + TAIL.format(name=name))
        node = next(n for n in lat.getNodes() if n.getName() == name)
        observed = node.getParam(param)
        if param == "kls":
            observed = observed[1]  # the quadrupole coefficient of knl={0, 1.5e-3}
        ok = abs(observed - expected) <= 1e-15 * abs(expected)
        print(
            f"[{label}] {param} observed {observed:.11g}, expected {expected:.11g} (the literal); reader message: {bool(out.strip())}"
        )
        if is_control:
            controls_ok &= ok
        else:
            bad |= not ok


    def ring_deck(fixed_point):
        """A 16-cell FODO ring with eight kicker correctors, as MAD-X SAVE writes it.

        With fixed_point False the text is the file MAD-X 5.09.03 SAVE writes for this ring (the
        companion script checks this). SAVE formats each value with %.10g, so the corrector kicks
        below 1e-4 appear as exponent literals; with fixed_point the same numbers are written
        without an exponent.
        """
        kicks = {
            "ch1": 3.7e-05,
            "ch2": -1.25e-05,
            "ch3": 2e-05,
            "ch4": -4.4e-05,
            "cv1": 1.5e-05,
            "cv2": -2.75e-05,
            "cv3": 6.1e-06,
            "cv4": -3.3e-05,
        }
        parents = {
            "qf": "qf: quadrupole,l:=lq ,k1:=kqf ;",
            "qd": "qd: quadrupole,l:=lq ,k1:=kqd ;",
            "mb": "mb: sbend,l:=lb ,angle:=2 * pi / ( 2 * ncell ) ;",
            "ch": "ch: kicker,l:= 0;",
            "cv": "cv: kicker,l:= 0;",
        }

        def lit(v):
            s = f"{v:.10g}"
            return format(decimal.Decimal(s), "f") if fixed_point else s

        defs = ["lq = 0.5;", "kqf = 0.2851;", "lb = 2;", "ncell = 16;", "kqd = -0.2851;"]
        seq = ["ring: sequence, l = 160;"]
        for i in range(16):
            s0 = 10 * i
            cell = [
                (0.25, f"qf{i}: qf"),
                (2.5, f"mb{i}a: mb"),
                (5.0, f"qd{i}: qd"),
                (7.5, f"mb{i}b: mb"),
            ]
            if i % 4 == 0:
                k = i // 4 + 1
                cell += [
                    (0.75, f"ch{k}: ch,hkick:= {lit(kicks[f'ch{k}'])}"),
                    (5.75, f"cv{k}: cv,vkick:= {lit(kicks[f'cv{k}'])}"),
                ]
            for at, text in sorted(cell):
                name, rest = text.split(": ")
                parent, _, attr = rest.partition(",")
                if parent in parents:  # SAVE defines each parent just before its first child
                    defs.append(parents.pop(parent))
                defs.append(f"{text};")
                seq.append(f"{name}: {parent}, at = {at + s0:g}" + (f",{attr};" if attr else ";"))
        return "\n".join(defs + seq + ["endsequence;"]) + "\n"


    def one_pass(text):
        lat, out = read(text)
        b = Bunch()
        b.mass(MASS)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        lat.trackBunch(b)
        kx = next(n.getParam("kx") for n in lat.getNodes() if n.getName() == "ch2")
        messages = [ln for ln in out.splitlines() if ln.strip()]
        return b.x(0), b.y(0), kx, lat.getLength(), messages


    saved = ring_deck(False)
    fixed = ring_deck(True)
    assert re.search(r"hkick:= -1\.25e-05", saved) and "e-0" not in fixed
    xs, ys, kxs, ls, ms = one_pass(saved)
    xf, yf, kxf, lf, mf = one_pass(fixed)
    # Both decks hold the same numbers, so they must give the same lattice and the same orbit.
    print(
        f"[ring, SAVE form] ch2 hkick observed {kxs:.4e}, expected -1.2500e-05; length {ls:.3f} m; reader messages: {len(ms)}"
    )
    print(
        f"[ring, SAVE form] one pass x observed {xs * 1e3:.4f} mm, y {ys * 1e3:.4f} mm; expected x {xf * 1e3:.4f} mm, y {yf * 1e3:.4f} mm (fixed-point deck)"
    )
    print(
        f"[control, ring, fixed-point literals] ch2 hkick {kxf:.4e}; length {lf:.3f} m; one pass x {xf * 1e3:.4f} mm, y {yf * 1e3:.4f} mm"
    )
    bad |= abs(xs - xf) > 1e-9
    controls_ok &= abs(kxf + 1.25e-05) < 1e-20
    print("DEFECT REPRODUCED" if bad and controls_ok else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-10-madx.py"

    [Download PAR-10-madx.py](PAR-10-madx.py)

    ```python
    """PAR-10: MAD-X reads the exponent literals as written, and SAVE writes small strengths with an exponent

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script evaluates the attribute literals of the main reproducer in MAD-X. It then builds the 16-cell FODO ring with eight kicker correctors in MAD-X, saves it with SAVE, checks that the saved file is the SAVE-form deck the main reproducer reads, prints its corrector lines, and computes the one-pass orbit from the origin with TWISS. MAD-X gives each literal its written value and x = 1.156 mm, y = −0.9718 mm at the end of the pass.
    """

    import decimal
    import os
    import tempfile

    from cpymad.madx import Madx

    KICKS = {
        "ch1": 3.7e-05,
        "ch2": -1.25e-05,
        "ch3": 2e-05,
        "ch4": -4.4e-05,
        "cv1": 1.5e-05,
        "cv2": -2.75e-05,
        "cv3": 6.1e-06,
        "cv4": -3.3e-05,
    }


    def ring_deck(fixed_point):
        """The SAVE-form deck of the main reproducer (same code)."""
        parents = {
            "qf": "qf: quadrupole,l:=lq ,k1:=kqf ;",
            "qd": "qd: quadrupole,l:=lq ,k1:=kqd ;",
            "mb": "mb: sbend,l:=lb ,angle:=2 * pi / ( 2 * ncell ) ;",
            "ch": "ch: kicker,l:= 0;",
            "cv": "cv: kicker,l:= 0;",
        }

        def lit(v):
            s = f"{v:.10g}"
            return format(decimal.Decimal(s), "f") if fixed_point else s

        defs = ["lq = 0.5;", "kqf = 0.2851;", "lb = 2;", "ncell = 16;", "kqd = -0.2851;"]
        seq = ["ring: sequence, l = 160;"]
        for i in range(16):
            s0 = 10 * i
            cell = [
                (0.25, f"qf{i}: qf"),
                (2.5, f"mb{i}a: mb"),
                (5.0, f"qd{i}: qd"),
                (7.5, f"mb{i}b: mb"),
            ]
            if i % 4 == 0:
                k = i // 4 + 1
                cell += [
                    (0.75, f"ch{k}: ch,hkick:= {lit(KICKS[f'ch{k}'])}"),
                    (5.75, f"cv{k}: cv,vkick:= {lit(KICKS[f'cv{k}'])}"),
                ]
            for at, text in sorted(cell):
                name, rest = text.split(": ")
                parent, _, attr = rest.partition(",")
                if parent in parents:
                    defs.append(parents.pop(parent))
                defs.append(f"{text};")
                seq.append(f"{name}: {parent}, at = {at + s0:g}" + (f",{attr};" if attr else ";"))
        return "\n".join(defs + seq + ["endsequence;"]) + "\n"


    TAIL = "seq: sequence, l=10, refer=centre;\n{name}, at=2;\nendsequence;\n"
    LITERALS = (
        ("quadrupole k1=1.5e-3", "q: quadrupole, l=1, k1=1.5e-3;\n", "q", "k1"),
        ("quadrupole k1=0.25e-3", "q: quadrupole, l=1, k1=0.25e-3;\n", "q", "k1"),
        ("quadrupole k1:=1.2345678901e-05", "q: quadrupole, l=1, k1:=1.2345678901e-05;\n", "q", "k1"),
        (
            "SAVE form, kicker hkick:= -1.25e-05",
            "c: kicker, l:= 0;\nc1: c,hkick:= -1.25e-05;\n",
            "c1",
            "hkick",
        ),
        (
            "SAVE form, kicker vkick:= 3.7e-05",
            "c: kicker, l:= 0;\nc1: c,vkick:= 3.7e-05;\n",
            "c1",
            "vkick",
        ),
        ("sbend angle=1.5e-3", "b: sbend, l=1, angle=1.5e-3;\n", "b", "angle"),
    )
    for label, definition, name, attr in LITERALS:
        m = Madx(stdout=False)
        m.input("beam;\n" + definition + TAIL.format(name=name))
        m.use("seq")
        print(f"MAD-X [{label}] {attr} = {float(m.elements[name][attr]):.11g}")
        m.quit()

    m = Madx(stdout=False)
    m.input("""
    beam, particle=proton, energy=1.938272088;  ! total energy: kinetic 1 GeV
    ncell = 16; lb = 2; lq = 0.5;
    kqf = 0.2851; kqd = -0.2851;
    qf: quadrupole, l:=lq, k1:=kqf;
    qd: quadrupole, l:=lq, k1:=kqd;
    mb: sbend, l:=lb, angle:=2*pi/(2*ncell);
    ch: kicker, l=0;
    cv: kicker, l=0;
    """)
    places = []
    for i in range(16):
        s0 = 10.0 * i
        places += [
            (s0 + 0.25, f"qf{i}: qf, at={s0 + 0.25:.4f};"),
            (s0 + 2.5, f"mb{i}a: mb, at={s0 + 2.5:.4f};"),
            (s0 + 5.0, f"qd{i}: qd, at={s0 + 5.0:.4f};"),
            (s0 + 7.5, f"mb{i}b: mb, at={s0 + 7.5:.4f};"),
        ]
        if i % 4 == 0:
            k = i // 4 + 1
            places += [
                (s0 + 0.75, f"ch{k}: ch, at={s0 + 0.75:.4f};"),
                (s0 + 5.75, f"cv{k}: cv, at={s0 + 5.75:.4f};"),
            ]
    m.input(
        "\n".join(
            ["ring: sequence, l=160, refer=centre;"] + [t for _, t in sorted(places)] + ["endsequence;"]
        )
    )
    for name, v in KICKS.items():
        m.input(f"{name}->{'hkick' if name.startswith('ch') else 'vkick'} = {v!r};")
    m.use("ring")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ring.seq")
        m.input(f'save, sequence=ring, file="{path}";')
        with open(path) as f:
            saved = f.read()
    print("MAD-X SAVE output equals the main reproducer's SAVE-form deck:", saved == ring_deck(False))
    print(
        "MAD-X SAVE corrector definitions:",
        [ln for ln in saved.splitlines() if "kick" in ln and "at =" not in ln],
    )
    tw = m.twiss(sequence="ring", betx=10, bety=10, x=0, px=0, y=0, py=0)  # one pass from the origin
    print(f"MAD-X one pass x {float(tw.x[-1]) * 1e3:.4f} mm, y {float(tw.y[-1]) * 1e3:.4f} mm")
    m.quit()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

