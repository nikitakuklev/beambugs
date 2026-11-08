---
title: "PAR-01: The MAD-X reader repeats the previous statement for every line without a semicolon"
hide:
  - toc
---

# PAR-01 — The MAD-X reader repeats the previous statement for every line without a semicolon

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P1 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`madx_parser.py#L312-L356`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L312-L356)<br>[`madx_parser.py#L431-L482`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L431-L482) |

`TEAPOT_Lattice.readMADX` keeps the text of the last line that ended a statement and adds it again for every following line without a ';': blank lines, comments indented by a space and the first line of a statement that continues on the next line. A blank line or an indented comment after a sequence placement therefore places that element twice. A thin multipole with knl₁ = 0.01 followed by a blank line kicks a particle at x = 1 mm by Δpx = −2.000×10⁻⁵ instead of −1.000×10⁻⁵, with no message; a 1 m quadrupole followed by an indented comment appears twice and a 10 m sequence becomes 11 m long. A blank first line or a statement split over two lines stops the read with an exception.

Run `python PAR-01.py`.

??? example "Reproducer: PAR-01.py"

    [Download PAR-01.py](PAR-01.py)

    ```python
    """PAR-01: The MAD-X reader repeats the previous statement for every line without a semicolon

    `TEAPOT_Lattice.readMADX` keeps the text of the last line that ended a statement and adds it again for every following line without a ';': blank lines, comments indented by a space and the first line of a statement that continues on the next line. A blank line or an indented comment after a sequence placement therefore places that element twice. A thin multipole with knl₁ = 0.01 followed by a blank line kicks a particle at x = 1 mm by Δpx = −2.000×10⁻⁵ instead of −1.000×10⁻⁵, with no message; a 1 m quadrupole followed by an indented comment appears twice and a 10 m sequence becomes 11 m long. A blank first line or a statement split over two lines stops the read with an exception.

    The script writes small MAD-X files to a temporary directory and reads each with `readMADX`. It tracks one proton at x = 1 mm through a thin multipole placed before a blank line, counts the quadrupoles and the lattice length when an indented comment follows the placement, and reports the exceptions raised by a blank first line and by a two-line definition. The control is the multipole file without the blank line; the expected values follow from MAD-X, where blank lines and comments are not statements and a statement ends at ';'.
    """

    import contextlib
    import io
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice


    def read(text):
        """Read a MAD-X file with this text; return the lattice and what the reader printed."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seq.madx")
            with open(path, "w") as f:
                f.write(text)
            lat = TEAPOT_Lattice()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                lat.readMADX(path, "ring")
        return lat, out.getvalue()


    def px_after(lat):
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
        lat.trackBunch(b)
        return b.xp(0)


    bad = False
    # Thin normal quadrupole term of a MULTIPOLE: dpx = -knl1 * x, here -0.01 * 1e-3 = -1e-5; drifts keep px.
    EXP_PX = -0.01 * 1e-3
    MP = "mp: multipole, knl={0, 0.01};\nring: sequence, l=10;\nmp, at=2;\n"
    for tag, text, is_control in (
        ("control, no blank line", MP + "endsequence;\n", True),
        ("blank line after thin multipole", MP + "\nendsequence;\n", False),
    ):
        lat, printed = read(text)
        n_mp = sum(1 for n in lat.getNodes() if n.getName() == "mp")
        px = px_after(lat)
        print(
            f"[{tag}] multipole nodes {n_mp}, expected 1; px observed {px:.6e}, expected {EXP_PX:.6e} (-knl1*x); "
            f"reader messages: {bool(printed.strip())}"
        )
        if not is_control:
            bad = bad or abs(px - EXP_PX) > 1e-15

    # Indented comment after a 1 m quadrupole placement; the sequence length L=10 m fixes the lattice length.
    lat, printed = read(
        "qa: quadrupole, l=1, k1=0.1;\nm1: marker;\nring: sequence, l=10;\nqa, at=2;\n  ! comment\nm1, at=6;\nendsequence;\n"
    )
    n_qa = [n.getName() for n in lat.getNodes()].count("qa")
    warned = "negative length" in printed
    print(
        f"[indented comment after quadrupole] qa nodes {n_qa}, expected 1; lattice length observed {lat.getLength():.6f} m, "
        f"expected 10.000000 m (sequence l); negative-length warning: {warned}"
    )
    bad = bad or n_qa != 1

    # A blank first line and a definition split over two lines: MAD-X reads both as a 10 m sequence with one quadrupole.
    for tag, text in (
        (
            "blank first line",
            "\nqa: quadrupole, l=1, k1=0.1;\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n",
        ),
        (
            "two-line definition",
            "m0: marker;\nqa: quadrupole, l=1,\n    k1=0.1;\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n",
        ),
    ):
        try:
            lat, _ = read(text)
            print(
                f"[{tag}] read ok, length {lat.getLength():.6f} m; expected a 10 m lattice with one quadrupole"
            )
        except Exception as e:  # noqa: BLE001 - any exception means the file was not read
            print(
                f"[{tag}] observed {type(e).__name__}: {e}; expected a 10 m lattice with one quadrupole"
            )
            bad = True
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-01-madx.py"

    [Download PAR-01-madx.py](PAR-01-madx.py)

    ```python
    """PAR-01: MAD-X reads the same files as one statement per ';'

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script writes the four files of the main reproducer, loads each into MAD-X with CALL, expands the sequence and prints its elements and length; for the multipole file it tracks one proton at x = 1 mm and prints px. MAD-X places each element once, gives px = −1.000×10⁻⁵ and reads the blank first line and the two-line definition.
    """

    import os
    import tempfile

    from cpymad.madx import Madx

    FILES = {
        "blank line after thin multipole": "mp: multipole, knl={0, 0.01};\nring: sequence, l=10;\nmp, at=2;\n\nendsequence;\n",
        "indented comment after quadrupole": "qa: quadrupole, l=1, k1=0.1;\nm1: marker;\nring: sequence, l=10;\nqa, at=2;\n  ! comment\nm1, at=6;\nendsequence;\n",
        "blank first line": "\nqa: quadrupole, l=1, k1=0.1;\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n",
        "two-line definition": "m0: marker;\nqa: quadrupole, l=1,\n    k1=0.1;\nring: sequence, l=10;\nqa, at=2;\nendsequence;\n",
    }
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for tag, text in FILES.items():
            path = os.path.join(d, "seq.madx")
            with open(path, "w") as f:
                f.write(text)
            m = Madx(stdout=False)
            m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
            m.call(path)
            m.input("use, sequence=ring;")
            names = [
                e.name
                for e in m.sequence.ring.expanded_elements
                if "$" not in e.name and not e.name.startswith("drift")
            ]
            line = f"[{tag}] elements {names}, sequence length {m.sequence.ring.length:.6f} m"
            ok = ok and len(names) == len(set(names))
            if tag.startswith("blank line after"):
                m.input("track, onepass=true, onetable=true; start, x=1e-3; run, turns=1; endtrack;")
                px = float(m.table.trackone.px[-1])
                line += f"; px after tracking {px:.6e} (-knl1*x = -1.000000e-05)"
                ok = ok and abs(px + 1e-5) < 1e-15
            print(line)
            m.quit()
    print("MAD-X places every element once" if ok else "MAD-X result differs from the expectation")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

