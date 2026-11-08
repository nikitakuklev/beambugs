---
title: "COL-13: addLongitudinalSpaceChargeNode and addImpedanceNode lengthen the lattice for a position outside it"
hide:
  - toc
---

# COL-13 — `addLongitudinalSpaceChargeNode` and `addImpedanceNode` lengthen the lattice for a position outside it

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [scLatticeModifications.py#L27-L62](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/space_charge/sc1d/scLatticeModifications.py#L27-L62)<br>[ImpedanceLatticeModifications.py#L31-L66](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/impedances/ImpedanceLatticeModifications.py#L31-L66) |

`addLongitudinalSpaceChargeNode(lattice, position, node)` and `addImpedanceNode(lattice, position, node)` search for the drift that contains `position` and, when none does, splice the node and a filler drift in front of the last node instead of raising an error. A position beyond the lattice end or below zero therefore lengthens the lattice silently. Inserting a zero-length node at 5 m into a lattice of three 1 m drifts gives a 6 m lattice (drifts of 1, 1 and 3 m, the node and the original last drift); at −0.5 m the lattice becomes 6.5 m long with the node at 2 m.

Run `python COL-13.py`.

??? example "Reproducer: COL-13.py"

    [Download COL-13.py](COL-13.py)

    ```python
    """COL-13: `addLongitudinalSpaceChargeNode` and `addImpedanceNode` lengthen the lattice for a position outside it

    `addLongitudinalSpaceChargeNode(lattice, position, node)` and `addImpedanceNode(lattice, position, node)` search for the drift that contains `position` and, when none does, splice the node and a filler drift in front of the last node instead of raising an error. A position beyond the lattice end or below zero therefore lengthens the lattice silently. Inserting a zero-length node at 5 m into a lattice of three 1 m drifts gives a 6 m lattice (drifts of 1, 1 and 3 m, the node and the original last drift); at −0.5 m the lattice becomes 6.5 m long with the node at 2 m.

    The script inserts a zero-length longitudinal space-charge node and a zero-length impedance node into fresh lattices of three 1 m TEAPOT drifts at 1.5 m, 5 m and −0.5 m, and prints the lattice length, the node's position and the node list. The insertion at 1.5 m, inside the lattice, is the control.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.impedances import LImpedance_Node, addImpedanceNode
    from orbit.space_charge.sc1d import SC1D_AccNode, addLongitudinalSpaceChargeNode
    from orbit.teapot import teapot


    def lattice3():
        lat = teapot.TEAPOT_Lattice("three drifts")
        for i in range(3):
            d = teapot.DriftTEAPOT(f"d{i}")
            d.setLength(1.0)
            lat.addNode(d)
        lat.initialize()
        return lat


    bad, ctl_ok = False, True
    helpers = (
        (addLongitudinalSpaceChargeNode, lambda: SC1D_AccNode(3.0, 3.0, 1, 1, 64)),
        (addImpedanceNode, lambda: LImpedance_Node(3.0, 1, 64)),
    )
    for helper, make in helpers:
        for pos in (1.5, 5.0, -0.5):
            lat = lattice3()
            node = make()
            helper(lat, pos, node)
            nodes = [(n.getName(), round(n.getLength(), 6)) for n in lat.getNodes()]
            at = lat.getNodePositionsDict()[node][0]
            length = lat.getLength()
            control = 0.0 <= pos <= 3.0
            # Expected: a zero-length insertion inside [0, 3] m keeps the 3 m total and places the node
            # at the requested position; a position outside the lattice should raise an error.
            expected = (
                "length 3 m, node at 1.5 m"
                if control
                else "an error (position outside the 3 m lattice)"
            )
            print(
                f"{helper.__name__} at {pos:+.1f} m{' (control)' if control else ''}: lattice length observed {length:.4f} m, "
                f"node at {at:.4f} m; expected {expected}; nodes {nodes}"
            )
            if control:
                ctl_ok &= abs(length - 3.0) < 1e-12 and abs(at - pos) < 1e-12
            else:
                bad |= abs(length - 3.0) > 1e-9
    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

