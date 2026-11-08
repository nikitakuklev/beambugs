---
title: "IO-03: Bunch.readBunch drops the last particle when the file does not end with a newline"
hide:
  - toc
---

# IO-03 — `Bunch.readBunch` drops the last particle when the file does not end with a newline

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [Bunch.cc#L1154-L1196](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L1154-L1196)<br>[Bunch.cc#L952-L971](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L952-L971)<br>[wrap_bunch.cc#L1100-L1132](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/wrap_bunch.cc#L1100-L1132) |

`Bunch.readBunch(path)` parses a coordinate line only while the stream has not reached
end-of-file, but reading a final line that lacks a terminating newline already sets that flag. The
last particle of such a file is therefore silently discarded. A three-row file `1 0 0 0 0 0`,
`2 0 0 0 0 0`, `3 0 0 0 0 0` without a final newline loads two particles (x = 1, 2) instead of three,
and a one-row file loads none. Files written by `dumpBunch` end with a newline and are unaffected.

Run `python IO-03.py`.

??? example "Reproducer: IO-03.py"

    [Download IO-03.py](IO-03.py)

    ```python
    """IO-03: `Bunch.readBunch` drops the last particle when the file does not end with a newline

    `Bunch.readBunch(path)` parses a coordinate line only while the stream has not reached end-of-file,
    but reading a final line that lacks a terminating newline already sets that flag. The last particle
    of such a file is therefore silently discarded. A three-row file `1 0 0 0 0 0`, `2 0 0 0 0 0`,
    `3 0 0 0 0 0` without a final newline loads two particles (x = 1, 2) instead of three, and a one-row
    file loads none. Files written by `dumpBunch` end with a newline and are unaffected.

    The script writes three six-column rows, and a single row, with and without a final newline, reads
    each file with `readBunch` into an empty bunch and compares the particle count and x values with the
    rows written. The files with a final newline are the control.
    """

    import os
    import tempfile

    from orbit.core.bunch import Bunch

    tmp = tempfile.mkdtemp()
    three = ["1 0 0 0 0 0", "2 0 0 0 0 0", "3 0 0 0 0 0"]
    cases = [
        ("three rows, no final newline", "\n".join(three), False),
        ("one row, no final newline", three[0], False),
        ("three rows, final newline (control)", "\n".join(three) + "\n", True),
        ("one row, final newline (control)", three[0] + "\n", True),
    ]
    ok = True
    for k, (label, text, control) in enumerate(cases):
        path = os.path.join(tmp, f"rows{k}.txt")
        with open(path, "w", newline="") as f:
            f.write(text)
        b = Bunch()
        b.readBunch(path)
        xs = [b.x(i) for i in range(b.getSize())]
        # Expected: one particle per data row, x taken from the first column of each row.
        exp = [float(line.split()[0]) for line in text.splitlines()]
        print(
            f"{label}: observed size {len(xs)} x={xs}; expected size {len(exp)} x={exp} (rows written)"
        )
        ok &= (xs == exp) if control else (xs == exp[:-1])

    print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

