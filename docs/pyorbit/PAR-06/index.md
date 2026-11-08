---
title: "PAR-06: A MAD-X INSTRUMENT with length does not drift the beam"
hide:
  - toc
---

# PAR-06 — A MAD-X INSTRUMENT with length does not drift the beam

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`madx_parser.py#L770-L771`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L770-L771)<br>[`teapot.py#L352-L360`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L352-L360) |

`readMADX` imports an INSTRUMENT as a marker, which the TEAPOT element factory builds as a node without a tracking map. An instrument of nonzero length keeps its length, so element positions and the lattice length are right, but particles cross it without moving. In MAD-X an INSTRUMENT acts on the beam as a drift of its length. In a 10 m sequence holding a 2 m instrument, a particle with px = 1×10⁻³ leaves at x = 8.000×10⁻³ m instead of 1.000×10⁻² m. Monitors and collimators with length are drifted correctly.

Run `python PAR-06.py`.

??? example "Reproducer: PAR-06.py"

    [Download PAR-06.py](PAR-06.py)

    ```python
    """PAR-06: A MAD-X INSTRUMENT with length does not drift the beam

    `readMADX` imports an INSTRUMENT as a marker, which the TEAPOT element factory builds as a node without a tracking map. An instrument of nonzero length keeps its length, so element positions and the lattice length are right, but particles cross it without moving. In MAD-X an INSTRUMENT acts on the beam as a drift of its length. In a 10 m sequence holding a 2 m instrument, a particle with px = 1×10⁻³ leaves at x = 8.000×10⁻³ m instead of 1.000×10⁻² m. Monitors and collimators with length are drifted correctly.

    The script reads 10 m MAD-X sequences holding one 2 m element with `readMADX`, prints the element's node type and the lattice length, and tracks one proton with px = 1×10⁻³ through the sequence. The expected x is L·px for the full 10 m, the result of a straight sequence of drifts at zero energy offset. The controls replace the instrument with a drift, a monitor and a collimator of the same length.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice

    EXPECTED = 10.0 * 1e-3  # field-free 10 m: x = L * px at dE = 0
    bad = False
    for typ in ("instrument", "drift", "monitor", "collimator"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seq.madx")
            with open(path, "w") as f:
                f.write(f"el: {typ}, l=2.0;\nring: sequence, l=10;\nel, at=5;\nendsequence;\n")
            lat = TEAPOT_Lattice()
            lat.readMADX(path, "ring")
        types = [n.getType() for n in lat.getNodes() if n.getName() == "el"]
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(0.0, 1e-3, 0.0, 0.0, 0.0, 0.0)
        lat.trackBunch(b)
        label = typ if typ == "instrument" else f"control, {typ}"
        print(
            f"[{label}] node type {types}, lattice length {lat.getLength():.3f} m; "
            f"x observed {b.x(0):.6e} m, expected {EXPECTED:.6e} m (10 m drift with px = 1e-3)"
        )
        if typ == "instrument":
            bad = abs(b.x(0) - EXPECTED) > 1e-12
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-06-madx.py"

    [Download PAR-06-madx.py](PAR-06-madx.py)

    ```python
    """PAR-06: MAD-X drifts the beam through the same instrument

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script loads the instrument sequence of the main reproducer and a drift control into MAD-X, tracks one proton with px = 1×10⁻³ and prints x at the end. MAD-X gives 1.000001×10⁻² m for both, its exact drift L·px/√(1 − px²); the 5×10⁻⁷ relative difference from L·px is the drift's square-root term.
    """

    import os
    import tempfile

    from cpymad.madx import Madx

    with tempfile.TemporaryDirectory() as d:
        for typ in ("instrument", "drift"):
            path = os.path.join(d, "seq.madx")
            with open(path, "w") as f:
                f.write(f"el: {typ}, l=2.0;\nring: sequence, l=10;\nel, at=5;\nendsequence;\n")
            m = Madx(stdout=False)
            m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
            m.call(path)
            m.input("use, sequence=ring;")
            m.input("track, onepass=true, onetable=true; start, px=1e-3; run, turns=1; endtrack;")
            print(f"MAD-X [{typ}] x at the end {float(m.table.trackone.x[-1]):.6e} m")
            m.quit()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

