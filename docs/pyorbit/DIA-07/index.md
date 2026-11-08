---
title: "DIA-07: addTeapotStatLatsNodeSet and addTeapotMomentsNodeSet label each record with the node entrance but record the state after the node's first part"
hide:
  - toc
---

# DIA-07 — `addTeapotStatLatsNodeSet` and `addTeapotMomentsNodeSet` label each record with the node entrance but record the state after the node's first part

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`diagnosticsLatticeModifications.py#L71-L103`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/diagnosticsLatticeModifications.py#L71-L103)<br>[`AccNode.py#L171-L198`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L171-L198)<br>[`AccNode.py#L328-L347`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccNode.py#L328-L347)<br>[`diagnosticsLatticeModifications.py#L66-L68`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/diagnosticsLatticeModifications.py#L66-L68) |

`addTeapotStatLatsNodeSet(lattice, filename)` and `addTeapotMomentsNodeSet` document one diagnostic "at start of each node" and label each record with the node's entrance position, but they attach the diagnostic as a body child with the default placement, after body part 0. A single-part element is therefore recorded at its exit, a multi-part element after its first part. In a lattice of two 1 m drifts entered with β = 1 m and α = 0, the records labelled s = 0 and s = 1 m report β = 2.000 m and 5.000 m, the values at s = 1 m and 2 m, instead of 1.000 m and 2.000 m.

Run `python DIA-07.py`.

??? example "Reproducer: DIA-07.py"

    [Download DIA-07.py](DIA-07.py)

    ```python
    """DIA-07: `addTeapotStatLatsNodeSet` and `addTeapotMomentsNodeSet` label each record with the node entrance but record the state after the node's first part

    `addTeapotStatLatsNodeSet(lattice, filename)` and `addTeapotMomentsNodeSet` document one diagnostic
    "at start of each node" and label each record with the node's entrance position, but they attach the
    diagnostic as a body child with the default placement, after body part 0. A single-part element is
    therefore recorded at its exit, a multi-part element after its first part. In a lattice of two 1 m
    drifts entered with β = 1 m and α = 0, the records labelled s = 0 and s = 1 m report β = 2.000 m and
    5.000 m, the values at s = 1 m and 2 m, instead of 1.000 m and 2.000 m.

    The script tracks an eight-particle bunch with β = 1 m, α = 0 and an energy spread of ±1 eV
    (so the dispersion terms vanish) through TEAPOT drift lattices with the statistics node set and
    compares each record's βx with the drift optics β(s) = β₀ + s²/β₀ at its labelled position: two
    1 m drifts, and a 2 m drift in four parts followed by a 1 m drift. A moments node set on the
    two-drift lattice is checked through its synchronous-particle time column. The control attaches
    the same statistics members at each node's entrance.
    """

    import itertools
    import os
    import tempfile

    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.diagnostics import addTeapotMomentsNodeSet, addTeapotStatLatsNodeSet
    from orbit.diagnostics.TeapotDiagnosticsNode import TeapotStatLatsNodeSetMember
    from orbit.lattice import AccNode
    from orbit.teapot import DriftTEAPOT, TEAPOT_Lattice

    TMP = tempfile.mkdtemp()


    def lattice(parts):
        lat = TEAPOT_Lattice()
        for k, (length, n) in enumerate(parts):
            d = DriftTEAPOT(f"d{k}")
            d.setLength(length)
            d.setnParts(n)
            lat.addNode(d)
        lat.initialize()
        return lat


    def bunch():
        b = Bunch()
        b.getSyncParticle().kinEnergy(1.0)
        # x = +-1e-4, x' = +-1e-4 (all sign pairs): <x^2> = <x'^2>, <x x'> = 0 -> beta0 = 1 m, alpha0 = 0.
        for sx, sxp, se in itertools.product((1, -1), repeat=3):
            b.addParticle(sx * 1e-4, sxp * 1e-4, sx * 1e-4, sxp * 1e-4, 0.0, se * 1e-9)
        return b


    def beta_drift(s):
        # Drift: beta(s) = beta0 - 2 alpha0 s + gamma0 s^2 with beta0 = 1 m, alpha0 = 0, gamma0 = 1/beta0.
        return 1.0 + s * s


    def entrance_set(lat, path):
        f = open(path, "w")  # noqa: SIM115 - the members write to it during tracking; closed after
        for node in lat.getNodes():
            m = TeapotStatLatsNodeSetMember(f)
            m.setPosition(lat.getNodePositionsDict()[node][0])
            m.setLatticeLength(lat.getLength())
            node.addChildNode(m, AccNode.ENTRANCE, 0, AccNode.BEFORE)
        return f


    def records(label, parts, attach):
        lat = lattice(parts)
        path = os.path.join(TMP, f"{label}.txt")
        handle = attach(lat, path)
        lat.trackBunch(bunch())
        (handle._file if hasattr(handle, "_file") else handle).close()
        with open(path) as f:
            rows = [[float(v) for v in line.split()] for line in f]
        out = []
        for r in rows:
            s, beta = r[0], r[4]
            print(
                f"{label}: record labelled s = {s:g} m: beta_x observed {beta:.6g}; expected {beta_drift(s):.6g}"
            )
            out.append((s, beta))
        return out


    set_2 = records("node set, two 1 m drifts", [(1.0, 1), (1.0, 1)], addTeapotStatLatsNodeSet)
    set_4 = records(
        "node set, 2 m drift in 4 parts + 1 m drift", [(2.0, 4), (1.0, 1)], addTeapotStatLatsNodeSet
    )
    ctl = records("entrance members (control), two 1 m drifts", [(1.0, 1), (1.0, 1)], entrance_set)

    lat = lattice([(1.0, 1), (1.0, 1)])
    path = os.path.join(TMP, "moments.txt")
    handle = addTeapotMomentsNodeSet(lat, path, 2, False, False)
    lat.trackBunch(bunch())
    handle._file.close()
    mom = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    for line in lines:
        f = line.split()
        s, frac = float(f[0]), float(f[1])
        # Column 2 is the synchronous-particle time divided by the lattice transit time, so at the
        # labelled position it equals s / L with L = 2 m.
        print(
            f"moments node set: record labelled s = {s:g} m: time fraction observed {frac:.6g}; expected {s / 2.0:.6g}"
        )
        mom.append((s, frac))

    shifted = all(abs(b - beta_drift(s + 1.0)) < 1e-9 for s, b in set_2)
    part0 = abs(set_4[0][1] - beta_drift(0.5)) < 1e-9
    mom_shift = all(abs(f - (s + 1.0) / 2.0) < 1e-9 for s, f in mom)
    ctl_ok = all(abs(b - beta_drift(s)) < 1e-9 for s, b in ctl)
    print("DEFECT REPRODUCED" if shifted and part0 and mom_shift and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

