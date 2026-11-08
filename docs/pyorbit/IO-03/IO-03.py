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
