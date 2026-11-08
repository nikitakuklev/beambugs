---
title: "LAT-03: Default tracking contexts persist across calls and lattices, so path_length and transport-matrix positions accumulate"
hide:
  - toc
---

# LAT-03 — Default tracking contexts persist across calls and lattices, so `path_length` and transport-matrix positions accumulate

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [AccLattice.py#L248-L263](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccLattice.py#L248-L263)<br>[AccNode.py#L307-L316](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L307-L316)<br>[AccNodeBunchTracker.py#L15-L29](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNodeBunchTracker.py#L15-L29)<br>[teapot.py#L127-L149](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L127-L149)<br>[MATRIX_Lattice.py#L472-L486](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L472-L486)<br>[LinacAccLatticeLib.py#L142-L163](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccLatticeLib.py#L142-L163)<br>[LinacAccNodes.py#L113-L127](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L113-L127)<br>[LinacAccLatticeLib.py#L165-L173](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccLatticeLib.py#L165-L173)<br>[LinacTransportMatrixGenNodes.py#L218-L250](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacTransportMatrixGenNodes.py#L218-L250)<br>[apertures_additions_lib.py#L18-L76](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/apertures_additions_lib.py#L18-L76)<br>[#L79-L123](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/apertures_additions_lib.py#L79-L123)<br>[#L126-L149](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/apertures_additions_lib.py#L126-L149)<br>[#L152-L197](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/apertures_additions_lib.py#L152-L197)<br>[#L200-L215](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/apertures_additions_lib.py#L200-L215) |

Seven tracking methods declare the context argument as `paramsDict={}`: `AccLattice.trackActions`, `AccNode.trackActions`, `AccNodeBunchTracker.trackBunch`, `TEAPOT_Lattice.trackBunch`, `MATRIX_Lattice.trackBunch`, `LinacAccLattice.trackActions` and `BaseLinacNode.trackDesignBunch`. Python creates each default dictionary once, so every call that omits the context continues from the `path_length` the previous call left, on any lattice. The linac transport-matrix controller positions its nodes through such a call: after a 3 m lattice, the matrix nodes of a 1 m lattice are placed at 3.0 and 3.5 m instead of 0.0 and 0.5 m. The linac aperture helpers' `aprtNodes=[]` default accumulates returned aperture nodes across lattices in the same way. The ordering of the matrix nodes and the matrices themselves are unaffected.

Run `python LAT-03.py`.

??? example "Reproducer: LAT-03.py"

    [Download LAT-03.py](LAT-03.py)

    ```python
    """LAT-03: Default tracking contexts persist across calls and lattices, so `path_length` and transport-matrix positions accumulate

    Seven tracking methods declare the context argument as `paramsDict={}`: `AccLattice.trackActions`, `AccNode.trackActions`, `AccNodeBunchTracker.trackBunch`, `TEAPOT_Lattice.trackBunch`, `MATRIX_Lattice.trackBunch`, `LinacAccLattice.trackActions` and `BaseLinacNode.trackDesignBunch`. Python creates each default dictionary once, so every call that omits the context continues from the `path_length` the previous call left, on any lattice. The linac transport-matrix controller positions its nodes through such a call: after a 3 m lattice, the matrix nodes of a 1 m lattice are placed at 3.0 and 3.5 m instead of 0.0 and 0.5 m. The linac aperture helpers' `aprtNodes=[]` default accumulates returned aperture nodes across lattices in the same way. The ordering of the matrix nodes and the matrices themselves are unaffected.

    The script builds two drift-only linac lattices (1 m + 2 m, then 0.5 m + 0.5 m) in one process, adds transport-matrix nodes at the drift entrances with a new controller per lattice, then an exit node on the second lattice, and compares the positions with the sums of the preceding drift lengths. It repeats a default-context `AccLattice.trackActions` call on a 3 m lattice and adds drift apertures to two fresh 2 m linac lattices. The control repeats the base-lattice calls with an explicit empty context.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.lattice import AccActionsContainer, AccLattice, AccNode
    from orbit.py_linac.lattice import Drift, LinacAccLattice, LinacTrMatricesController, Sequence
    from orbit.py_linac.lattice_modifications import Add_drift_apertures_to_lattice


    def linac(name, lengths):
        lat = LinacAccLattice(name)
        seq = Sequence(name + "_seq")
        seq.setLinacAccLattice(lat)
        for i, length in enumerate(lengths):
            d = Drift(f"{name}_d{i}")
            d.setLength(length)
            seq.addNode(d)
            lat.addNode(d)
        lat.initialize()
        return lat


    def starts(lengths):
        # Entrance of drift i within its own lattice: the sum of the preceding drift lengths.
        return [sum(lengths[:i], 0.0) for i in range(len(lengths))]


    def fmt(values):
        return "[" + ", ".join(f"{v:.4g}" for v in values) + "]"


    reproduced = False

    # Transport-matrix nodes at the drift entrances, one controller per lattice.
    for name, lengths in (("first", [1.0, 2.0]), ("second", [0.5, 0.5])):
        lat = linac(name, lengths)
        ctrl = LinacTrMatricesController()
        ctrl.addTrMatrixGenNodesAtEntrance(lat, lat.getNodes())
        obs = [n.getPosition() for n in ctrl.getNodes()]
        exp = starts(lengths)
        print(f"{name} lattice, entrance matrix nodes [m]: observed {fmt(obs)}, expected {fmt(exp)}")
        reproduced |= obs != exp

    # A second call of the same controller on the second lattice (exit of its last drift) repositions all nodes.
    ctrl.addTrMatrixGenNodesAtExit(lat, [lat.getNodes()[-1]])
    obs = [n.getPosition() for n in ctrl.getNodes()]
    exp = starts([0.5, 0.5]) + [1.0]  # entrances 0 and 0.5 m, exit of the 1 m lattice
    print(
        f"second lattice, entrance and exit matrix nodes after a second call [m]: observed {fmt(obs)}, expected {fmt(exp)}"
    )
    reproduced |= obs != exp

    # Base lattice: exit path_length of three default-context passes over one 3 m node.
    base = AccLattice("base")
    node = AccNode("n")
    node.setLength(3.0)
    base.addNode(node)
    base.initialize()
    for label, context in (("default context", None), ("control, explicit {} each call", "fresh")):
        seen = []
        acts = AccActionsContainer()
        acts.addAction(lambda d, seen=seen: seen.append(d["path_length"]), AccNode.EXIT)
        for _ in range(3):
            if context is None:
                base.trackActions(acts)
            else:
                base.trackActions(acts, {})
        # Each pass traverses the 3 m lattice once from path_length 0.
        print(
            f"AccLattice.trackActions, {label}, exit path_length of 3 passes [m]: observed {fmt(seen)}, expected [3, 3, 3]"
        )
        if context is None:
            reproduced |= seen != [3.0, 3.0, 3.0]
        elif seen != [3.0, 3.0, 3.0]:
            raise RuntimeError(
                "control case failed: an explicit context does not start at path_length 0"
            )

    # Aperture helper with its default list: apertures every 0.5 m from 0 to 2 m on two 1 m drifts (one part each)
    # place one aperture at each drift entrance, two per lattice.
    for name in ("apertures_a", "apertures_b"):
        lat = linac(name, [1.0, 1.0])
        returned = Add_drift_apertures_to_lattice(lat, 0.0, 2.0, 0.5, 0.02)
        own = sum(1 for n in returned if n.getName().startswith(name))
        print(
            f"Add_drift_apertures_to_lattice on {name}: returned {len(returned)} nodes, expected {own} (nodes added to {name})"
        )
        reproduced |= len(returned) != own

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

