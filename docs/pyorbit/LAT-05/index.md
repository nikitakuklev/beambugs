---
title: "LAT-05: AccNode guard messages raise TypeError while being built, and a failed setLength leaves the node inconsistent"
hide:
  - toc
---

# LAT-05 — `AccNode` guard messages raise `TypeError` while being built, and a failed `setLength` leaves the node inconsistent

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [AccNode.py#L77-L90](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L77-L90)<br>[AccNode.py#L115-L134](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L115-L134)<br>[AccNode.py#L171-L194](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L171-L194)<br>[AccNode.py#L51-L69](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L51-L69) |

Two guards of the base lattice node build their error message by concatenating an integer or a tuple to a string. Total `setLength()` on a node that has body children, and `addChildNode()` with an invalid placement code, therefore raise `TypeError: can only concatenate str (not "int") to str` instead of printing the guard's explanation and stopping through `orbitFinalize`. `setLength` stores the new total before the guard runs, so a caller that catches the `TypeError` keeps a node whose length is 2.0 m while its parts still sum to 1.0 m. The sibling guard in `setnParts()` builds its message correctly and stops as intended.

Run `python LAT-05.py`.

??? example "Reproducer: LAT-05.py"

    [Download LAT-05.py](LAT-05.py)

    ```python
    """LAT-05: `AccNode` guard messages raise `TypeError` while being built, and a failed `setLength` leaves the node inconsistent

    Two guards of the base lattice node build their error message by concatenating an integer or a tuple to a string. Total `setLength()` on a node that has body children, and `addChildNode()` with an invalid placement code, therefore raise `TypeError: can only concatenate str (not "int") to str` instead of printing the guard's explanation and stopping through `orbitFinalize`. `setLength` stores the new total before the guard runs, so a caller that catches the `TypeError` keeps a node whose length is 2.0 m while its parts still sum to 1.0 m. The sibling guard in `setnParts()` builds its message correctly and stops as intended.

    The script triggers both guards on pure-Python lattice nodes, reports the exception each raises, and prints the node's total and part lengths after the caught `setLength` failure. The expected result is the guard's stop, `SystemExit(1)` from `orbitFinalize`. The control is `setnParts(3)` on a node with a body child, whose guard reaches `orbitFinalize`.
    """

    import contextlib
    import io
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.lattice import AccNode


    def attempt(f):
        """Run f with the guard's printout suppressed; return the exception it raised."""
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                f()
                return "nothing raised"
            except (SystemExit, TypeError) as e:
                return f"{type(e).__name__}({e})"


    def node_with_body_child():
        n = AccNode("parent")
        n.setLength(1.0)
        n.addChildNode(AccNode("child"), AccNode.BODY)
        return n


    EXPECTED = "SystemExit(1) from orbitFinalize with the guard message"
    reproduced = False
    cases = [
        ("setLength(2.0) with a body child", lambda: node_with_body_child().setLength(2.0), False),
        ("addChildNode(child, place=7)", lambda: AccNode("p").addChildNode(AccNode("c"), 7), False),
        ("control, setnParts(3) with a body child", lambda: node_with_body_child().setnParts(3), True),
    ]
    for label, f, control in cases:
        obs = attempt(f)
        print(f"{label}: observed {obs}; expected {EXPECTED}")
        if control and obs != "SystemExit(1)":
            raise RuntimeError("control case failed: setnParts no longer stops through orbitFinalize")
        reproduced |= not control and obs.startswith("TypeError")

    # State after the caught TypeError: the total was assigned before the guard, the parts were not rebuilt.
    n = node_with_body_child()
    attempt(lambda: n.setLength(2.0))
    parts = sum(n.getLength(i) for i in range(n.getnParts()))
    print(
        f"after the caught setLength(2.0): total length {n.getLength():.4g} m, sum of part lengths {parts:.4g} m; expected equal"
    )
    reproduced |= n.getLength() != parts
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

