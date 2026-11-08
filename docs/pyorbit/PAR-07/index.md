---
title: "PAR-07: MAD-X FROM= is measured from the reference point of the named element, not its centre"
hide:
  - toc
---

# PAR-07 — MAD-X FROM= is measured from the reference point of the named element, not its centre

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`madx_parser.py#L459-L467`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L459-L467)<br>[`madx_parser.py#L404-L408`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L404-L408)<br>[`test_madx_parser.py#L9-L22`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/tests/py/orbit/parsers/test_madx_parser.py#L9-L22) |

In a MAD-X sequence, `AT=a, FROM=name` places an element at the distance a from the centre of the element `name`. `readMADX` adds a to the stored placement position of `name`, which is its entry or its exit when the sequence uses REFER=ENTRY or REFER=EXIT, so an element placed with FROM lands half the named element's length too early or too late. A marker at `AT=3, FROM=qa` after a 2 m quadrupole lands at 4 m instead of 5 m with REFER=ENTRY and at 6 m with REFER=EXIT. With the default REFER=CENTRE the position is right. For a name placed several times the reader measures from the latest earlier placement, where MAD-X uses the first.

Run `python PAR-07.py`.

??? example "Reproducer: PAR-07.py"

    [Download PAR-07.py](PAR-07.py)

    ```python
    """PAR-07: MAD-X FROM= is measured from the reference point of the named element, not its centre

    In a MAD-X sequence, `AT=a, FROM=name` places an element at the distance a from the centre of the element `name`. `readMADX` adds a to the stored placement position of `name`, which is its entry or its exit when the sequence uses REFER=ENTRY or REFER=EXIT, so an element placed with FROM lands half the named element's length too early or too late. A marker at `AT=3, FROM=qa` after a 2 m quadrupole lands at 4 m instead of 5 m with REFER=ENTRY and at 6 m with REFER=EXIT. With the default REFER=CENTRE the position is right. For a name placed several times the reader measures from the latest earlier placement, where MAD-X uses the first.

    The script reads short MAD-X sequences with `readMADX` and prints the marker's position, the sum of the node lengths before it. In the REFER cases the quadrupole's centre is at 2 m, so the expected marker position is 2 + 3 = 5 m; the REFER=CENTRE sequence is the control. The last case places a 1 m quadrupole at 2 m and 12 m and a marker at `AT=11, FROM=qa`, which MAD-X puts at 2 + 11 = 13 m.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.teapot import TEAPOT_Lattice

    # A 2 m quadrupole whose centre is at 2 m: AT=1 is its entry, AT=2 its centre, AT=3 its exit.
    CASES = (
        ("refer=entry", "ring: sequence, l=10, refer=entry;\nqa, at=1;\n", 5.0, "clean"),
        ("refer=exit", "ring: sequence, l=10, refer=exit;\nqa, at=3;\n", 5.0, "clean"),
        ("control, refer=centre", "ring: sequence, l=10;\nqa, at=2;\n", 5.0, "control"),
    )
    DEF = "qa: quadrupole, l=2, k1=0.1;\nm1: marker;\n"
    FROM = "m1, at=3, from=qa;\nendsequence;\n"
    REPEATED = "qa: quadrupole, l=1, k1=0.1;\nm1: marker;\nring: sequence, l=30;\nqa, at=2;\nqa, at=12;\nm1, at=11, from=qa;\nendsequence;\n"


    def marker_position(text):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seq.madx")
            with open(path, "w") as f:
                f.write(text)
            lat = TEAPOT_Lattice()
            lat.readMADX(path, "ring")
        s = 0.0
        for node in lat.getNodes():
            if node.getName() == "m1":
                return s
            s += node.getLength()
        raise RuntimeError("marker m1 not in the lattice")


    bad = False
    for tag, seq, expected, kind in CASES:
        s = marker_position(DEF + seq + FROM)
        print(
            f"[{tag}] marker position observed {s:.6f} m, expected {expected:.6f} m (centre of qa at 2 m + AT 3 m)"
        )
        if kind == "clean":
            bad = bad or abs(s - expected) > 1e-12
    s = marker_position(REPEATED)
    print(
        f"[repeated name] marker position observed {s:.6f} m, MAD-X 13.000000 m (first qa centred at 2 m + AT 11 m)"
    )
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-07-madx.py"

    [Download PAR-07-madx.py](PAR-07-madx.py)

    ```python
    """PAR-07: MAD-X positions of the same markers, and its verdict on the upstream repeated-name test file

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script expands the sequences of the main reproducer in MAD-X and prints the marker position; MAD-X puts it at 5 m for every REFER value and at 13 m in the repeated-name case. It then loads the upstream test input `repeated_element.lat` (a BPM at 1 m and 2 m, a probe at `AT=0.5, FROM=bpm`, which the upstream test expects at 2.5 m), which MAD-X refuses with a negative drift because it measures from the first BPM.
    """

    import os
    import tempfile

    from cpymad.madx import Madx

    DEF = "qa: quadrupole, l=2, k1=0.1;\nm1: marker;\n"
    FROM = "m1, at=3, from=qa;\nendsequence;\n"
    FILES = {
        "refer=entry": DEF + "ring: sequence, l=10, refer=entry;\nqa, at=1;\n" + FROM,
        "refer=exit": DEF + "ring: sequence, l=10, refer=exit;\nqa, at=3;\n" + FROM,
        "refer=centre": DEF + "ring: sequence, l=10;\nqa, at=2;\n" + FROM,
        "repeated name": "qa: quadrupole, l=1, k1=0.1;\nm1: marker;\nring: sequence, l=30;\nqa, at=2;\nqa, at=12;\nm1, at=11, from=qa;\nendsequence;\n",
    }
    UPSTREAM_TEST = "bpm: monitor;\nprobe: marker;\nring: sequence, l=3.0;\nbpm, at=1.0;\nbpm, at=2.0;\nprobe, at=0.5, from=bpm;\nendsequence;\n"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.madx")
        for tag, text in FILES.items():
            with open(path, "w") as f:
                f.write(text)
            m = Madx(stdout=False)
            m.input("beam, particle=proton, energy=2;")
            m.call(path)
            m.input("use, sequence=ring;")
            pos = [e.position for e in m.sequence.ring.expanded_elements if e.name.startswith("m1")]
            print(f"MAD-X [{tag}] marker position {pos[0]:.6f} m")
            m.quit()
        with open(path, "w") as f:
            f.write(UPSTREAM_TEST)
        log = os.path.join(d, "madx.log")
        with open(log, "w") as out:
            m = Madx(stdout=out, stderr=out)
            m.input("beam, particle=proton, energy=2;")
            m.call(path)
            try:
                m.input("use, sequence=ring;")
                print(
                    "MAD-X [repeated_element.lat]",
                    [(e.name, e.position) for e in m.sequence.ring.expanded_elements],
                )
            except RuntimeError:
                pass
            m.quit()
        with open(log) as f:
            print(
                "MAD-X [repeated_element.lat]", [line.strip() for line in f if "negative drift" in line]
            )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small class="bb-scope">source reading</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source reading |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

