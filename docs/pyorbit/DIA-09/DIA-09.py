"""DIA-09: `closeStatLats` and `closeMoments` close an unused file handle and leave the records unwritten

`TeapotStatLatsNode(filename)` and `TeapotMomentsNode(filename, order)` open `filename` twice: their
`StatLats` or `Moments` helper opens it for appending and writes every record, and the node then
opens the same path for writing in a second handle that nothing writes. `closeStatLats()` and
`closeMoments()` close only the node's empty handle, so the records stay in the helper's buffer, and
the file is empty, or ends at a buffer boundary inside a record, until the helper is
garbage-collected or the interpreter exits. After three records and the close call both files have 0
bytes; closing the helper's handle then writes the three records of each.

The script tracks a four-particle bunch three times through a `TeapotStatLatsNode` and through a
`TeapotMomentsNode`, calls the node's close method and counts the complete records in the file.
The control then closes the helper's own handle and counts again.
"""

import os
import tempfile

from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
from orbit.diagnostics import TeapotMomentsNode, TeapotStatLatsNode

TMP = tempfile.mkdtemp()


def bunch():
    b = Bunch()
    b.getSyncParticle().kinEnergy(1.0)
    for sx, sxp, se in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
        b.addParticle(sx * 1e-3, sxp * 1e-4, sx * 2e-3, sxp * 1e-4, 0.0, se * 1e-3)
    return b


def count(path):
    with open(path) as f:
        text = f.read()
    return os.path.getsize(path), text.count("\n")


ok = True
for label, make, close, helper in (
    ("TeapotStatLatsNode", lambda p: TeapotStatLatsNode(p), "closeStatLats", lambda n: n.statlats),
    ("TeapotMomentsNode", lambda p: TeapotMomentsNode(p, 2), "closeMoments", lambda n: n.moments),
):
    path = os.path.join(TMP, f"{label}.txt")
    node = make(path)
    b = bunch()
    for _ in range(3):
        node.track({"bunch": b})
    getattr(node, close)()
    size_c, rec_c = count(path)
    getattr(helper(node), close)()  # the helper's handle holds the records
    size_f, rec_f = count(path)
    # Expected: after the close method returns, the file holds every record written: three.
    print(f"{label}: after {close}(): {size_c} bytes, {rec_c} records; expected 3 records")
    print(f"{label}: after closing the helper's handle (control): {size_f} bytes, {rec_f} records")
    ok &= rec_c < 3 and rec_f == 3

print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")
