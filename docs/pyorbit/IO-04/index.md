---
title: "IO-04: Bunch.readBunch applies the last row's column count to every row of a read chunk"
hide:
  - toc
---

# IO-04 — `Bunch.readBunch` applies the last row's column count to every row of a read chunk

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [Bunch.cc#L1165-L1181](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L1165-L1181)<br>[Bunch.cc#L1224-L1228](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L1224-L1228)<br>[Bunch.cc#L1141-L1144](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L1141-L1144) |

`Bunch.readBunch(path)` accepts data rows with six coordinates or with six coordinates
plus the particle-attribute columns, but decides whether to copy attribute columns from the column
count of the last row parsed in each 1000-row chunk. In a file that mixes both widths, a chunk that
ends with a six-column row discards the attribute values of all its full-width rows, and a chunk that
ends with a full-width row gives its six-column rows stale values left in the read buffer by an
earlier read. Appending one six-column row to a `dumpBunch` file whose particles have `macrosize` 5
and 7 reloads both with `macrosize` 0. Files with a single row width are read correctly.

Run `python IO-04.py`.

??? example "Reproducer: IO-04.py"

    [Download IO-04.py](IO-04.py)

    ```python
    """IO-04: `Bunch.readBunch` applies the last row's column count to every row of a read chunk

    `Bunch.readBunch(path)` accepts data rows with six coordinates or with six coordinates plus the
    particle-attribute columns, but decides whether to copy attribute columns from the column count of
    the last row parsed in each 1000-row chunk. In a file that mixes both widths, a chunk that ends with
    a six-column row discards the attribute values of all its full-width rows, and a chunk that ends with
    a full-width row gives its six-column rows stale values left in the read buffer by an earlier read.
    Appending one six-column row to a `dumpBunch` file whose particles have `macrosize` 5 and 7 reloads
    both with `macrosize` 0. Files with a single row width are read correctly.

    The script dumps two particles with `macrosize` 5 and 7, appends a six-column row, and separately
    writes the same header with a six-column row placed before the full-width rows, read right after a
    file whose first particle has `macrosize` 99. Each file is read into an empty bunch and every row's
    (x, macrosize) is compared with the columns written, a six-column row keeping the attribute's zero
    initialization. Controls read the unmodified dump and a file of six-column rows only.
    """

    import os
    import tempfile

    from orbit.core.bunch import Bunch

    tmp = tempfile.mkdtemp()


    def dump(name, xs, ws):
        b = Bunch()
        b.addPartAttr("macrosize")
        for i, (x, w) in enumerate(zip(xs, ws)):
            b.addParticle(x, 0.0, 0.0, 0.0, 0.0, 0.0)
            b.partAttrValue("macrosize", i, 0, w)
        path = os.path.join(tmp, name)
        b.dumpBunch(path)
        with open(path) as f:
            return path, f.read()


    def write(name, text):
        path = os.path.join(tmp, name)
        with open(path, "w") as f:
            f.write(text)
        return path


    def load(path):
        c = Bunch()
        c.readBunch(path)
        return [(c.x(i), c.partAttrValue("macrosize", i, 0)) for i in range(c.getSize())]


    base, text = dump("base.txt", [1.0, 2.0], [5.0, 7.0])
    header = "".join(line + "\n" for line in text.splitlines() if line.startswith("%"))
    full_rows = "".join(line + "\n" for line in text.splitlines() if not line.startswith("%"))
    end_six = write("end_six.txt", text + "3 0 0 0 0 0\n")
    start_six = write("start_six.txt", header + "0.5 0 0 0 0 0\n" + full_rows)
    six_only = write("six_only.txt", header + "1 0 0 0 0 0\n2 0 0 0 0 0\n")
    primer, _ = dump("primer.txt", [9.0], [99.0])

    # Expected: a full-width row keeps its own seventh column; a six-column row has no attribute column
    # and keeps the attribute's initial value 0.
    results = []
    for label, path, exp in (
        ("full, full, six-column rows", end_six, [(1.0, 5.0), (2.0, 7.0), (3.0, 0.0)]),
        (
            "six-column, full, full rows after a macrosize-99 read",
            start_six,
            [(0.5, 0.0), (1.0, 5.0), (2.0, 7.0)],
        ),
        ("unmodified dump (control)", base, [(1.0, 5.0), (2.0, 7.0)]),
        ("six-column rows only (control)", six_only, [(1.0, 0.0), (2.0, 0.0)]),
    ):
        if path == start_six:
            load(primer)  # leaves 99 in the attribute slot of buffer row 0
        obs = load(path)
        print(f"{label}: observed {obs}; expected {exp} (columns written per row)")
        results.append(obs == exp)

    print("DEFECT REPRODUCED" if results == [False, False, True, True] else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed<br><small class="bb-scope">stale values for six-column rows in a chunk ending with a full-width row</small> | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

