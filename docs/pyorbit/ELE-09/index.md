---
title: "ELE-09: Linac longitudinal-displacement errors raise NameError; behind it the exit node repeats the entrance shift and the Gaussian setter assigns σ"
hide:
  - toc
---

# ELE-09 — Linac longitudinal-displacement errors raise NameError; behind it the exit node repeats the entrance shift and the Gaussian setter assigns σ

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Elements and apertures |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`errors_modifications_lib.py#L207-L211`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/errors_modifications_lib.py#L207-L211)<br>[`ErrorNodesAndControllersLib.py#L104-L122`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/errors/ErrorNodesAndControllersLib.py#L104-L122)<br>[`ErrorNodesAndControllersLib.py#L398-L408`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/errors/ErrorNodesAndControllersLib.py#L398-L408)<br>[`errors_modifications_lib.py#L241-L251`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice_modifications/errors_modifications_lib.py#L241-L251)<br>[`ErrorNode.py#L872-L884`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/errors/ErrorNode.py#L872-L884) |

The linac longitudinal-misalignment path cannot be used. `LongitudinalDisplacementNodesModification()` raises NameError in its constructor, which assigns an undefined `shift_length`, and the error nodes of `ErrorCntrlLongitudinalDisplacement` raise NameError on their first tracked bunch because they pass an undefined `bunc` to `LongDisplacement`. A 1 cm shift should drift the beam by +1 cm at the entrance and −1 cm at the exit, moving a particle with x′ = 1 mrad by 10.00 μm and back. Behind these errors the controller gives the exit node the same +1 cm, so the two drifts add to 2 cm instead of cancelling, and `setGaussDistributedShiftLength` assigns σ to every node instead of the drawn values.

Run `python ELE-09.py`.

??? example "Reproducer: ELE-09.py"

    [Download ELE-09.py](ELE-09.py)

    ```python
    """ELE-09: Linac longitudinal-displacement errors raise NameError; behind it the exit node repeats the entrance shift and the Gaussian setter assigns σ

    The linac longitudinal-misalignment path cannot be used. `LongitudinalDisplacementNodesModification()` raises NameError
    in its constructor, which assigns an undefined `shift_length`, and the error nodes of
    `ErrorCntrlLongitudinalDisplacement` raise NameError on their first tracked bunch because they pass an undefined `bunc`
    to `LongDisplacement`. A 1 cm shift should drift the beam by +1 cm at the entrance and −1 cm at the exit, moving a
    particle with x′ = 1 mrad by 10.00 μm and back. Behind these errors the controller gives the exit node the same +1 cm, so
    the two drifts add to 2 cm instead of cancelling, and `setGaussDistributedShiftLength` assigns σ to every node instead of
    the drawn values.

    The script calls the constructor and tracks a bunch through the controller's entrance and exit nodes and prints the
    exceptions. It then reads the shift lengths the controller hands to its two nodes, tracks through both with the `bunc`
    name corrected in memory, and compares with the TEAPOT error pair (+ds, −ds); and it calls the Gaussian setter on four
    nodes with the constructor's missing assignment supplied. The control is the coordinate-displacement controller of the
    same module.
    """

    import os
    import random

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.core.error_base import LongDisplacement  # ty: ignore[unresolved-import]
    from orbit.py_linac.errors import (
        ErrorCntrlCoordDisplacement,
        ErrorCntrlLongitudinalDisplacement,
        ErrorLongitudinalDisplacementNode,
    )
    from orbit.py_linac.lattice import Quad
    from orbit.py_linac.lattice_modifications import LongitudinalDisplacementNodesModification
    from orbit.py_linac.lattice_modifications.errors_modifications_lib import ErrorForNodesModification

    DS = 0.01  # 1 cm shift


    def bunch():
        b = Bunch()
        b.mass(0.93827208816)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(0.4)
        b.addParticle(0.0, 1e-3, 0.0, 0.0, 0.0, 0.0)  # x' = 1 mrad
        return b


    def attempt(label, call, expected):
        try:
            call()
            print(f"[{label}] observed: no exception; expected {expected}")
            return False
        except NameError as e:
            print(f"[{label}] observed NameError: {e}; expected {expected}")
            return True


    reproduced = attempt(
        "LongitudinalDisplacementNodesModification()",
        lambda: LongitudinalDisplacementNodesModification("m"),
        "a constructed object",
    )
    cntrl = ErrorCntrlLongitudinalDisplacement("c")
    cntrl.setShiftLength(DS)
    for name, node, sign in (
        ("entrance", cntrl.getEntanceNode(), 1),
        ("exit", cntrl.getExitNode(), -1),
    ):
        # A drift of ds at x' = 1 mrad moves x by ds x' (TEAPOT drift at dE = 0): +10 um at the entrance, -10 um at the exit.
        expected = f"a {sign:+d} cm drift, x = {sign * DS * 1e-3:+.3e} m"
        reproduced &= attempt(
            f"controller {name} node, shift 1 cm",
            lambda node=node: node.track({"bunch": bunch()}),
            expected,
        )

    # The node docstring: the controller returns shift_length for the entrance node and -shift_length for the exit one.
    ent, ext = cntrl.getEntanceNodeShiftLength(), cntrl.getExitNodeShiftLength()
    print(
        f"[controller shift lengths] entrance {ent:+.4f} m, exit {ext:+.4f} m; expected {DS:+.4f} m, {-DS:+.4f} m"
    )
    reproduced &= ext == ent


    def corrected_track(self, paramsDict):
        LongDisplacement(paramsDict["bunch"], self.error_controller_params_func())


    # the one-character fix of the `bunc` name, applied in memory
    ErrorLongitudinalDisplacementNode.track = corrected_track
    b = bunch()
    cntrl.getEntanceNode().track({"bunch": b})
    x_entrance = b.x(0)
    cntrl.getExitNode().track({"bunch": b})
    ref = bunch()
    LongDisplacement(ref, DS)
    LongDisplacement(
        ref, -DS
    )  # the TEAPOT LongDisp error pair: longdisplacement(ds), longdisplacement(-ds)
    print(
        f"[entrance + exit with `bunc` corrected] x after entrance {x_entrance:.6e} m, expected 1.000000e-05 m; x after exit "
        f"{b.x(0):.6e} m, expected {ref.x(0):.1e} m (TEAPOT pair +ds, -ds)"
    )
    reproduced &= abs(x_entrance - 1e-5) < 1e-15 and abs(b.x(0) - 2e-5) < 1e-15

    # Gaussian setter, with the constructor's missing assignment supplied: each node must get its own draw within 3 sigma.
    m = LongitudinalDisplacementNodesModification.__new__(LongitudinalDisplacementNodesModification)
    ErrorForNodesModification.__init__(m, "m", "LongitudinalDisplacementNodesModification")
    m.shift_length = 0.0
    m.addLatticeNodes([Quad(f"q{i}") for i in range(4)])
    random.seed(20261107)
    m.setGaussDistributedShiftLength(DS)
    values = [c.getShiftLength() for c in m.getErrorControllers()]
    print(
        f"[setGaussDistributedShiftLength(sigma = 0.01), 4 nodes] assigned {values}; expected 4 distinct draws, |s| <= 0.03 m"
    )
    reproduced &= values == [DS] * 4

    # Control: the coordinate-displacement controller returns +dx for the entrance node and -dx for the exit node.
    cc = ErrorCntrlCoordDisplacement("cc")
    cc.setDisplacementParameter("dx", 1e-3)
    b = bunch()
    cc.getEntanceNode().track({"bunch": b})
    x_in = b.x(0)
    cc.getExitNode().track({"bunch": b})
    print(
        f"[control, coordinate displacement dx = 1 mm] x after entrance {x_in:.6e} m, expected 1.000000e-03 m; after exit {b.x(0):.1e} m, expected 0"
    )
    reproduced &= abs(x_in - 1e-3) < 1e-15 and abs(b.x(0)) < 1e-15
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small>error-node NameError</small> | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed<br><small>NameErrors, exit-node sign and Gaussian setter</small> | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

