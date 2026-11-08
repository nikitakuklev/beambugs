---
title: "LAT-01: trackBunch leaves its tracking action in a caller's container after an exception, so the next call repeats every map"
hide:
  - toc
---

# LAT-01 — `trackBunch` leaves its tracking action in a caller's container after an exception, so the next call repeats every map

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [teapot.py#L127-L149](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L127-L149)<br>[LinacAccLatticeLib.py#L165-L204](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccLatticeLib.py#L165-L204)<br>[AccNodeBunchTracker.py#L15-L29](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNodeBunchTracker.py#L15-L29)<br>[MATRIX_Lattice.py#L472-L486](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L472-L486)<br>[LinacAccNodes.py#L113-L127](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L113-L127)<br>[AccActionsContainer.py#L28-L55](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccActionsContainer.py#L28-L55) |

When an action or a node raises inside `TEAPOT_Lattice.trackBunch`, `LinacAccLattice.trackBunch` or `trackDesignBunch`, or `AccNodeBunchTracker.trackBunch`, the temporary BODY action that applies each node's map stays in the caller's action container, because the method removes it only after a normal return. If the caller catches the exception and tracks again with that container, every node's map runs once more per earlier failure. For a 1 m drift and x′ = 10⁻³, x after one caught failure is 2.000×10⁻³ m instead of 1.000×10⁻³ m, and after three it is 4.000×10⁻³ m. Calls that let the method create its own container are unaffected.

Run `python LAT-01.py`.

??? example "Reproducer: LAT-01.py"

    [Download LAT-01.py](LAT-01.py)

    ```python
    """LAT-01: `trackBunch` leaves its tracking action in a caller's container after an exception, so the next call repeats every map

    When an action or a node raises inside `TEAPOT_Lattice.trackBunch`, `LinacAccLattice.trackBunch` or `trackDesignBunch`, or `AccNodeBunchTracker.trackBunch`, the temporary BODY action that applies each node's map stays in the caller's action container, because the method removes it only after a normal return. If the caller catches the exception and tracks again with that container, every node's map runs once more per earlier failure. For a 1 m drift and x′ = 10⁻³, x after one caught failure is 2.000×10⁻³ m instead of 1.000×10⁻³ m, and after three it is 4.000×10⁻³ m. Calls that let the method create its own container are unaffected.

    The script tracks one particle (x = 0, x′ = 10⁻³, on energy) through a 1 m TEAPOT drift, a 1 m linac drift and a bare `AccNodeBunchTracker` with a caller-supplied container whose EXIT diagnostic raises on the first calls; it catches the exceptions, then tracks a fresh particle with the same container. It also raises from the TEAPOT node itself and a `KeyboardInterrupt` from the diagnostic, as an interrupted notebook cell does. The expected x is the drift map x + L·x′; the control tracks with a fresh container after the failures.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.lattice import AccActionsContainer, AccNodeBunchTracker
    from orbit.py_linac.lattice import Drift, LinacAccLattice, Sequence
    from orbit.teapot import DriftTEAPOT, TEAPOT_Lattice

    L, XP = 1.0, 1.0e-3
    # For dE = 0 both drift maps give x_out = x + L*x' exactly, so one pass from x = 0 gives L*x'.
    EXPECTED_X = L * XP


    def particle():
        b = Bunch()
        b.getSyncParticle().kinEnergy(1.0)  # GeV
        b.addParticle(0.0, XP, 0.0, 0.0, 0.0, 0.0)
        return b


    def failing_container(exc_type):
        """Container whose EXIT diagnostic raises while state['on'] is true."""
        state = {"on": True}

        def diagnostic(params):
            if state["on"]:
                raise exc_type("diagnostic failed")

        acts = AccActionsContainer("caller")
        acts.addAction(diagnostic, AccActionsContainer.EXIT)
        return acts, state


    def n_body(acts):
        return len(acts.getActions(AccActionsContainer.BODY))


    def teapot_drift(node_cls=DriftTEAPOT):
        lat = TEAPOT_Lattice("ring")
        d = node_cls("d")
        d.setLength(L)
        lat.addNode(d)
        lat.initialize()
        return lat


    def linac_drift():
        lat = LinacAccLattice("linac")
        seq = Sequence("seq")
        seq.setLinacAccLattice(lat)
        d = Drift("d")
        d.setLength(L)
        seq.addNode(d)
        lat.addNode(d)
        lat.initialize()
        return lat


    reproduced = False


    def report(label, failures, acts, x, expected=EXPECTED_X, control=False):
        global reproduced
        print(
            f"{label}: BODY actions left after {failures} caught failure(s) {n_body(acts)}, expected 0; "
            f"x after reuse {x:.3e} m, expected x + L*x' = {expected:.3e} m"
        )
        if not control and (n_body(acts) != 0 or x != expected):
            reproduced = True
        if control and x != expected:
            raise RuntimeError("control case failed: the drift map itself differs from x + L*x'")


    def run_failures(track, acts, n, exc_type):
        for _ in range(n):
            try:
                track(particle(), acts)
            except exc_type:
                pass


    # TEAPOT lattice, one and three caught failures of a diagnostic.
    lat = teapot_drift()
    for n in (1, 3):
        acts, state = failing_container(RuntimeError)
        run_failures(lambda b, a: lat.trackBunch(b, {}, a), acts, n, RuntimeError)
        state["on"] = False
        b = particle()
        lat.trackBunch(b, {}, acts)
        report(f"TEAPOT_Lattice.trackBunch, diagnostic raises {n}x", n, acts, b.x(0))

    # TEAPOT lattice, KeyboardInterrupt from the diagnostic (interrupted cell, then rerun).
    acts, state = failing_container(KeyboardInterrupt)
    run_failures(lambda b, a: lat.trackBunch(b, {}, a), acts, 1, KeyboardInterrupt)
    state["on"] = False
    b = particle()
    lat.trackBunch(b, {}, acts)
    report("TEAPOT_Lattice.trackBunch, KeyboardInterrupt", 1, acts, b.x(0))


    # TEAPOT lattice, the node's own track() raises once.
    class FailingDrift(DriftTEAPOT):
        fail = True

        def track(self, paramsDict):
            if FailingDrift.fail:
                raise RuntimeError("node failed")
            DriftTEAPOT.track(self, paramsDict)


    lat_f = teapot_drift(FailingDrift)
    acts = AccActionsContainer("caller")
    try:
        lat_f.trackBunch(particle(), {}, acts)
    except RuntimeError:
        pass
    FailingDrift.fail = False
    b = particle()
    lat_f.trackBunch(b, {}, acts)
    report("TEAPOT_Lattice.trackBunch, node raises", 1, acts, b.x(0))

    # Linac lattice: trackBunch and trackDesignBunch each leave one action behind.
    lin = linac_drift()
    acts, state = failing_container(RuntimeError)
    run_failures(lambda b, a: lin.trackBunch(b, {}, a), acts, 1, RuntimeError)
    run_failures(lambda b, a: lin.trackDesignBunch(b, {}, a), acts, 1, RuntimeError)
    state["on"] = False
    b = particle()
    lin.trackBunch(b, {}, acts)
    # The stale trackDesign action moves the particle as well: Drift.trackDesign calls track().
    report("LinacAccLattice.trackBunch + trackDesignBunch", 2, acts, b.x(0))


    # AccNodeBunchTracker: count track() calls instead of applying a map.
    class Counter(AccNodeBunchTracker):
        calls = 0

        def track(self, paramsDict):
            Counter.calls += 1


    node = Counter("node")
    node.setLength(L)
    acts, state = failing_container(RuntimeError)
    run_failures(lambda b, a: node.trackBunch(b, {}, a), acts, 1, RuntimeError)
    state["on"] = False
    Counter.calls = 0
    node.trackBunch(particle(), {}, acts)
    print(
        f"AccNodeBunchTracker.trackBunch: track() calls on reuse after 1 caught failure {Counter.calls}, expected 1"
    )
    reproduced |= Counter.calls != 1

    # Control: the TEAPOT lattice used above, tracked with a fresh container, still makes one pass.
    b = particle()
    fresh = AccActionsContainer("fresh")
    lat.trackBunch(b, {}, fresh)
    report("control, fresh container", 0, fresh, b.x(0), control=True)

    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

