---
title: "OPT-08: A BaseMATRIX node with several parts applies its whole matrix in every part"
hide:
  - toc
---

# OPT-08 — A `BaseMATRIX` node with several parts applies its whole matrix in every part

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Optics |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`BaseMATRIX.py#L29-L34`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/BaseMATRIX.py#L29-L34)<br>[`AccNode.py#L328-L347`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L328-L347)<br>[`MATRIX_Lattice.py#L53-L71`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L53-L71)<br>[`MATRIX_Lattice.py#L255-L258`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L255-L258)<br>[`teapot_matrix_lattice.py#L60-L68`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot_matrix_lattice.py#L60-L68)<br>[`LinacRfGapNodes.py#L105-L110`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L105-L110) |

`BaseMATRIX.track` applies the node's full 7×7 transfer matrix M each time it runs, and lattice tracking runs it once per body part. A matrix node given N parts, for example to host child nodes between parts, therefore transports particles with Mᴺ, while `MATRIX_Lattice` builds its one-turn matrix, Twiss functions and tunes from M once. A 1 m drift matrix node with 2 parts moves a particle at x = 1 mm, x′ = 1 mrad to x = 3.000 mm, where the lattice's one-turn matrix and a TEAPOT drift give 2.000 mm. Matrix nodes built by `TEAPOT_MATRIX_Lattice` have one part and are unaffected.

Run `python OPT-08.py`.

??? example "Reproducer: OPT-08.py"

    [Download OPT-08.py](OPT-08.py)

    ```python
    """OPT-08: A `BaseMATRIX` node with several parts applies its whole matrix in every part

    `BaseMATRIX.track` applies the node's full 7×7 transfer matrix M each time it runs, and lattice tracking runs it once per body part. A matrix node given N parts, for example to host child nodes between parts, therefore transports particles with Mᴺ, while `MATRIX_Lattice` builds its one-turn matrix, Twiss functions and tunes from M once. A 1 m drift matrix node with 2 parts moves a particle at x = 1 mm, x′ = 1 mrad to x = 3.000 mm, where the lattice's one-turn matrix and a TEAPOT drift give 2.000 mm. Matrix nodes built by `TEAPOT_MATRIX_Lattice` have one part and are unaffected.

    The script builds a `MATRIX_Lattice` of one `BaseMATRIX` node holding the 1 m drift matrix, gives it 1, 2, 3 and 5 parts, and tracks a particle with `trackBunch`. Each tracked x is compared with the lattice's own one-turn matrix applied to the particle and with a 1 m `DriftTEAPOT`; one part is the control.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.matrix_lattice import BaseMATRIX, MATRIX_Lattice
    from orbit.teapot import DriftTEAPOT, TEAPOT_Lattice

    X0, XP0, L = 1e-3, 1e-3, 1.0  # start x [m], x' [rad], drift length [m]


    def bunch():
        b = Bunch()
        b.mass(0.93827208816)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(X0, XP0, 0.0, 0.0, 0.0, 0.0)
        return b


    # Reference: a TEAPOT drift; for dE = 0 it gives x = x0 + L*x' exactly.
    teapot = TEAPOT_Lattice("drift")
    teapot.addNode(DriftTEAPOT("d", L))
    teapot.initialize()
    b = bunch()
    teapot.trackBunch(b)
    x_teapot = b.x(0)

    defect = True
    for n in (1, 2, 3, 5):
        node = BaseMATRIX("m")
        node.setLength(L)
        m = node.getMatrix()
        m.unit()
        m.set(0, 1, L)  # drift matrix: x += L*x'
        node.setnParts(n)
        lat = MATRIX_Lattice("lat")
        lat.addNode(node)
        lat.initialize()
        b = bunch()
        lat.trackBunch(b)
        one_turn = lat.getOneTurnMatrix()
        # the lattice's one-turn matrix applied to the particle
        x_matrix = X0 + one_turn.get(0, 1) * XP0
        label = "control, 1 part" if n == 1 else f"{n} parts"
        print(
            f"[{label}] tracked x {b.x(0) * 1e3:.6f} mm; expected {x_matrix * 1e3:.6f} mm "
            f"(one-turn matrix, M01 = {one_turn.get(0, 1):.6f} m) and {x_teapot * 1e3:.6f} mm (TEAPOT drift)"
        )
        ok = abs(b.x(0) - x_matrix) < 1e-15 and abs(b.x(0) - x_teapot) < 1e-15
        defect = defect and (ok if n == 1 else not ok)
    print("DEFECT REPRODUCED" if defect else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

