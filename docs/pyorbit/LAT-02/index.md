---
title: "LAT-02: TEAPOT_Lattice.getSubLattice rewrites the ring length of RF cavities in the original lattice"
hide:
  - toc
---

# LAT-02 — `TEAPOT_Lattice.getSubLattice` rewrites the ring length of RF cavities in the original lattice

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [teapot.py#L110-L125](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L110-L125)<br>[AccLattice.py#L218-L233](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/lattice/AccLattice.py#L218-L233)<br>[teapot.py#L1309-L1326](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1309-L1326) |

`TEAPOT_Lattice.getSubLattice` builds a new lattice that shares its node objects with the original and initializes it, and `TEAPOT_Lattice.initialize` writes the lattice length into every `RingRFTEAPOT` node as its ring length C. Taking a sublattice that contains a ring RF cavity therefore changes the cavity in the original ring, whose RF kick qV·sin(2πhz/C) then uses the sublattice length. For a 100 m ring with a cavity after the first 10 m drift, `getSubLattice(0, 1)` changes the cavity's ring length from 100 m to 10 m, and the kick at z = 1 m grows from 6.279×10⁻⁵ GeV to 5.878×10⁻⁴ GeV. Calling `initialize()` on the original ring restores it.

Run `python LAT-02.py`.

??? example "Reproducer: LAT-02.py"

    [Download LAT-02.py](LAT-02.py)

    ```python
    """LAT-02: `TEAPOT_Lattice.getSubLattice` rewrites the ring length of RF cavities in the original lattice

    `TEAPOT_Lattice.getSubLattice` builds a new lattice that shares its node objects with the original and initializes it, and `TEAPOT_Lattice.initialize` writes the lattice length into every `RingRFTEAPOT` node as its ring length C. Taking a sublattice that contains a ring RF cavity therefore changes the cavity in the original ring, whose RF kick qV·sin(2πhz/C) then uses the sublattice length. For a 100 m ring with a cavity after the first 10 m drift, `getSubLattice(0, 1)` changes the cavity's ring length from 100 m to 10 m, and the kick at z = 1 m grows from 6.279×10⁻⁵ GeV to 5.878×10⁻⁴ GeV. Calling `initialize()` on the original ring restores it.

    The script builds the 100 m ring with a single-harmonic 1 MV cavity (V = 10⁻³ GV, h = 1, zero phase), takes the sublattice of the first drift and the cavity, and tracks a proton at z = 1 m, dE = 0 through the original ring. It compares the cavity's ring length with the ring circumference and the energy kick with qV·sin(2πhz/C) for C = 100 m. The control tracks the same particle before the sublattice is taken.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.teapot import DriftTEAPOT, RingRFTEAPOT, TEAPOT_Lattice

    V, H, Z = 1.0e-3, 1, 1.0  # GV, harmonic, m

    ring = TEAPOT_Lattice("ring")
    rf = RingRFTEAPOT("rf")
    rf.addRF(H, V, 0.0)
    ring.addNode(DriftTEAPOT("d1", 10.0))
    ring.addNode(rf)
    ring.addNode(DriftTEAPOT("d2", 90.0))
    ring.initialize()
    C = ring.getLength()


    def kick():
        """Energy change of a proton at z = Z, dE = 0 after one pass through the original ring."""
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(0.0, 0.0, 0.0, 0.0, Z, 0.0)
        ring.trackBunch(b)
        return b.dE(0)


    # Ring RF kick for a particle with zero phase: dE = q V sin(2 pi h z / C); drifts do not change dE.
    expected_kick = V * math.sin(2.0 * math.pi * H * Z / C)

    control = kick()
    print(
        f"control, before getSubLattice: kick {control:.3e} GeV, expected qV sin(2 pi h z/C) = {expected_kick:.3e} GeV"
    )
    if abs(control - expected_kick) > 1e-12:
        raise RuntimeError("control case failed: the RF kick model differs from qV sin(2 pi h z/C)")

    sub = ring.getSubLattice(0, 1)
    ring_length = rf.getParam("ring_length")
    after = kick()
    print(f"sublattice length {sub.getLength():.4g} m")
    print(
        f"cavity ring_length after getSubLattice(0, 1): observed {ring_length:.4g} m, expected {C:.4g} m (ring circumference)"
    )
    print(
        f"kick in the original ring after getSubLattice: observed {after:.3e} GeV, expected {expected_kick:.3e} GeV"
    )
    reproduced = ring_length != C or abs(after - expected_kick) > 1e-12
    ring.initialize()
    print(
        f"cavity ring_length after ring.initialize(): {rf.getParam('ring_length'):.4g} m, expected {C:.4g} m"
    )
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

