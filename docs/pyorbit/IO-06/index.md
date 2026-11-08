---
title: "IO-06: load_bunch sets the synchronous-particle position to the saved momentum vector"
hide:
  - toc
---

# IO-06 — `load_bunch` sets the synchronous-particle position to the saved momentum vector

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [serialize.py#L194-L204](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L194-L204)<br>[serialize.py#L12-L29](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L12-L29)<br>[serialize.py#L355-L358](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/bunch_utils/serialize.py#L355-L358)<br>[SyncPart.cc#L133-L138](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SyncPart.cc#L133-L138)<br>[wrap_syncpart.cc#L70-L92](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/wrap_syncpart.cc#L70-L92)<br>[RungeKuttaTracker.cc#L155-L172](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/trackerrk4/RungeKuttaTracker.cc#L155-L172) |

`collect_bunch` stores the synchronous particle's momentum vector `pVector()` under
`sync_part["coords"]`, documented as the synchronous particle's coordinates, and `load_bunch` passes
that vector to the position setter `rVector()`, then rebuilds the momentum along z from the kinetic
energy. After a `save_bunch`/`load_bunch` round trip the reference particle therefore sits at the
numerical value of its momentum in metres, and any momentum direction other than z is lost. For a
1 GeV proton reference at the origin moving along z, the reloaded position is (0, 0, 1.696) m
instead of (0, 0, 0); a reference at (0.25, −0.5, 7.0) m tilted by 0.1 rad in x reloads at
(0.1693, 0, 1.688) m with its momentum along z. Kinetic energy and time are restored to double-precision rounding.

Run `python IO-06.py`.

??? example "Reproducer: IO-06.py"

    [Download IO-06.py](IO-06.py)

    ```python
    """IO-06: `load_bunch` sets the synchronous-particle position to the saved momentum vector

    `collect_bunch` stores the synchronous particle's momentum vector `pVector()` under
    `sync_part["coords"]`, documented as the synchronous particle's coordinates, and `load_bunch` passes
    that vector to the position setter `rVector()`, then rebuilds the momentum along z from the kinetic
    energy. After a `save_bunch`/`load_bunch` round trip the reference particle therefore sits at the
    numerical value of its momentum in metres, and any momentum direction other than z is lost. For a
    1 GeV proton reference at the origin moving along z, the reloaded position is (0, 0, 1.696) m
    instead of (0, 0, 0); a reference at (0.25, −0.5, 7.0) m tilted by 0.1 rad in x reloads at
    (0.1693, 0, 1.688) m with its momentum along z. Kinetic energy and time are restored to double-precision rounding.

    The script builds two serial bunches with a 1 GeV proton reference, one at the origin moving along
    z and one off the origin with its momentum tilted by 0.1 rad in x, collects each into a temporary
    directory, saves the collection with `save_bunch`, reloads it with `load_bunch` and compares the
    reloaded position and momentum vectors with those at save. Kinetic energy and time, which the
    round trip restores, are the control.
    """

    import math
    import tempfile

    from orbit.bunch_utils import collect_bunch, load_bunch, save_bunch
    from orbit.core.bunch import Bunch


    def fmt(v):
        return "(" + ", ".join(f"{c:.6g}" for c in v) + ")"


    def round_trip(position, tilt):
        b = Bunch()
        b.mass(0.93827208816)
        sp = b.getSyncParticle()
        sp.kinEnergy(1.0)
        p = sp.momentum()
        # Momentum of magnitude p tilted by `tilt` from z towards x.
        sp.pVector((p * math.sin(tilt), 0.0, p * math.cos(tilt)))
        sp.rVector(position)
        sp.time(3.0e-6)
        b.addParticle(1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0)
        saved = collect_bunch(b, tempfile.mkdtemp(), return_memmap=False)
        d = tempfile.mkdtemp()
        save_bunch(saved, d)
        c, _ = load_bunch(d)
        s2 = c.getSyncParticle()
        return sp, s2


    defect, control = True, True
    for label, position, tilt in (
        ("reference at the origin along z", (0.0, 0.0, 0.0), 0.0),
        ("reference at (0.25, -0.5, 7.0) m tilted 0.1 rad in x", (0.25, -0.5, 7.0), 0.1),
    ):
        sp, s2 = round_trip(position, tilt)
        # Expected: every saved reference quantity is restored unchanged.
        print(
            f"{label}: position observed {fmt(s2.rVector())} m; expected {fmt(sp.rVector())} m (position at save)"
        )
        print(
            f"{label}: momentum observed {fmt(s2.pVector())} GeV/c; expected {fmt(sp.pVector())} GeV/c (momentum at save)"
        )
        print(
            f"{label}: kinetic energy and time (control) observed {s2.kinEnergy()!r} GeV, {s2.time()!r} s; "
            f"expected {sp.kinEnergy()!r} GeV, {sp.time()!r} s"
        )
        defect &= s2.rVector() == sp.pVector()
        control &= abs(s2.kinEnergy() - sp.kinEnergy()) < 1e-15 and s2.time() == sp.time()
        if tilt:
            defect &= s2.pVector()[0] == 0.0 and sp.pVector()[0] != 0.0

    print("DEFECT REPRODUCED" if defect and control else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed<br><small class="bb-scope">lost momentum direction</small> | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

