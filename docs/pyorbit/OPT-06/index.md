---
title: "OPT-06: TEAPOT_MATRIX_Lattice.rebuild raises KeyError for quadrupoles, multipoles and bends"
hide:
  - toc
---

# OPT-06 — `TEAPOT_MATRIX_Lattice.rebuild` raises KeyError for quadrupoles, multipoles and bends

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Optics |
| Priority | P4 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot_matrix_lattice.py#L91-L118`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot_matrix_lattice.py#L91-L118)<br>[`teapot_matrix_lattice.py#L49-L86`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot_matrix_lattice.py#L49-L86)<br>[`AccNode.py#L307-L356`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L307-L356)<br>[`teapot.py#L904-L920`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L904-L920)<br>[`teapot.py#L753-L757`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L753-L757)<br>[`teapot.py#L1071-L1083`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1071-L1083) |

`TEAPOT_MATRIX_Lattice.rebuild(kin_energy)`, which remeasures the matrices at a new reference energy, tracks each recorded node with a parameter dictionary holding only the bunch and the node. The constructor also recorded the fringe-field child nodes, whose functions read `paramsDict["parentNode"]`, which only lattice traversal supplies, so rebuild raises KeyError: 'parentNode' at the first bend, or at the first quadrupole or multipole with a fringe field enabled, the default. A lattice of one quadrupole and one drift fails. A drift-only lattice, or a quadrupole with both fringe fields disabled, rebuilds and matches a fresh build at the new energy exactly.

Run `python OPT-06.py`.

??? example "Reproducer: OPT-06.py"

    [Download OPT-06.py](OPT-06.py)

    ```python
    """OPT-06: `TEAPOT_MATRIX_Lattice.rebuild` raises KeyError for quadrupoles, multipoles and bends

    `TEAPOT_MATRIX_Lattice.rebuild(kin_energy)`, which remeasures the matrices at a new reference
    energy, tracks each recorded node with a parameter dictionary holding only the bunch and the node.
    The constructor also recorded the fringe-field child nodes, whose functions read
    `paramsDict["parentNode"]`, which only lattice traversal supplies, so rebuild raises KeyError:
    'parentNode' at the first bend, or at the first quadrupole or multipole with a fringe field enabled,
    the default. A lattice of one quadrupole and one drift fails. A drift-only lattice, or a quadrupole
    with both fringe fields disabled, rebuilds and matches a fresh build at the new energy exactly.

    The script builds five two-element lattices (one element followed by a 1 m drift), constructs the
    matrix lattice at 1 GeV and calls `rebuild(2.0)`. The expected result is the one-turn matrix of a
    matrix lattice built fresh at 2 GeV. A drift-only lattice and a quadrupole with both fringe fields
    disabled are the controls.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.teapot import (
        BendTEAPOT,
        DriftTEAPOT,
        MultipoleTEAPOT,
        QuadTEAPOT,
        TEAPOT_Lattice,
        TEAPOT_MATRIX_Lattice,
    )


    def proton_bunch(kin_energy):
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(kin_energy)  # GeV
        return b


    def make_lattice(kind):
        lat = TEAPOT_Lattice("l")
        if kind == "quadrupole":
            lat.addNode(QuadTEAPOT("q", 0.5, 2, 1.0))
        elif kind == "quadrupole, fringe fields disabled":
            q = QuadTEAPOT("q", 0.5, 2, 1.0)
            q.setUsageFringeFieldIN(False)
            q.setUsageFringeFieldOUT(False)
            lat.addNode(q)
        elif kind == "sextupole multipole":
            lat.addNode(MultipoleTEAPOT("m", 0.3, 2, [2], [0.5], [0]))
        elif kind == "bend, fringe fields disabled":
            b = BendTEAPOT("b", 1.0, 2, theta=0.1)
            b.setUsageFringeFieldIN(False)
            b.setUsageFringeFieldOUT(False)
            lat.addNode(b)
        lat.addNode(DriftTEAPOT("d", 1.0))
        lat.initialize()
        return lat


    def matrix(ml):
        m = ml.getOneTurnMatrix()
        return [[m.get(i, j) for j in range(6)] for i in range(6)]


    def max_diff(a, b):
        return max(abs(x - y) for ra, rb in zip(a, b, strict=True) for x, y in zip(ra, rb, strict=True))


    bad = False
    for kind in (
        "drift only",
        "quadrupole, fringe fields disabled",
        "quadrupole",
        "sextupole multipole",
        "bend, fringe fields disabled",
    ):
        ml = TEAPOT_MATRIX_Lattice(make_lattice(kind), proton_bunch(1.0))
        fresh = matrix(TEAPOT_MATRIX_Lattice(make_lattice(kind), proton_bunch(2.0)))
        # the 1 GeV and 2 GeV matrices differ (M56 of the drift at least), so a rebuild that kept the
        # 1 GeV matrices would show here
        shift = max_diff(matrix(ml), fresh)
        try:
            ml.rebuild(2.0)
            diff = max_diff(matrix(ml), fresh)
            obs = f"ok, max |rebuilt - fresh| = {diff:.1e} (1 GeV vs 2 GeV: {shift:.1e})"
            if diff > 1e-12:
                bad = True
        except KeyError as e:
            obs = f"KeyError: {e}"
            bad = True
        print(
            f"[{kind}] rebuild(2.0): observed {obs}; expected the 6x6 one-turn matrix of a fresh 2 GeV build"
        )
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

