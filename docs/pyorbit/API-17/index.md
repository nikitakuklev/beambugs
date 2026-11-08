---
title: "API-17: Docstrings give MeV and volts for quantities the code takes in GeV and GV"
hide:
  - toc
---

# API-17 — Docstrings give MeV and volts for quantities the code takes in GeV and GV

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Bindings and build |
| Priority | P5 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [wrap_bunch.cc#L1211](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/wrap_bunch.cc#L1211)<br>[wrap_syncpart.cc#L553](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/wrap_syncpart.cc#L553)<br>[teapot.py#L1265-L1270](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L1265-L1270)<br>[Bunch.hh#L109-L122](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.hh#L109-L122)<br>[teapotbase.cc#L1519-L1565](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L1519-L1565) |

`Bunch.mass` is documented as "the mass of particle in MeV" and `SyncPart.kinEnergy` as "kinetic energy of the synchronous particle in MeV", but both are in GeV, as the native comments and the rigidity `B_Rho` assume. `RingRFTEAPOT` documents "Voltages are in Volts", while the ring RF kick adds q·V directly to energies in GeV, so its voltage is in GV. Following the RF docstring with V = 1000 for 1 kV gives an energy kick of 1000 GeV instead of 1.000×10⁻⁶ GeV.

Run `python API-17.py`.

??? example "Reproducer: API-17.py"

    [Download API-17.py](API-17.py)

    ```python
    """API-17: Docstrings give MeV and volts for quantities the code takes in GeV and GV

    `Bunch.mass` is documented as "the mass of particle in MeV" and `SyncPart.kinEnergy` as "kinetic energy of the synchronous particle in MeV", but both are in GeV, as the native comments and the rigidity `B_Rho` assume. `RingRFTEAPOT` documents "Voltages are in Volts", while the ring RF kick adds q·V directly to energies in GeV, so its voltage is in GV. Following the RF docstring with V = 1000 for 1 kV gives an energy kick of 1000 GeV instead of 1.000×10⁻⁶ GeV.

    The script prints the three docstrings, the default bunch mass, and the momentum and rigidity after `kinEnergy(1.0)`, which identify the units. It then tracks one proton through a 100 m ring with a harmonic-1 RF cavity of V = 1000 at zero synchronous phase, placed at z = C/4 where sin(2πz/C) = 1, and compares its energy kick with that of 1 kV.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.teapot import DriftTEAPOT, RingRFTEAPOT, TEAPOT_Lattice

    b = Bunch()
    print(f"Bunch.mass doc: {Bunch.mass.__doc__!r}")
    print(f"SyncPart.kinEnergy doc: {type(b.getSyncParticle()).kinEnergy.__doc__!r}")
    print(
        f"RingRFTEAPOT doc: {[ln.strip() for ln in (RingRFTEAPOT.__init__.__doc__ or '').splitlines() if 'olt' in ln]}"
    )
    # The proton rest mass is 938.272 MeV = 0.938272 GeV.
    print(f"default mass: observed {b.mass()}, expected 938.272 if in MeV, 0.938272 if in GeV")

    b.mass(0.938272)
    b.charge(1)
    s = b.getSyncParticle()
    s.kinEnergy(1.0)
    # p = sqrt(K (K + 2m)); with K = 1 GeV and m = 0.938272 GeV, p = 1.6960 GeV/c. Brho = p/(0.299792458 q) T m.
    p_gev = math.sqrt(1.0 * (1.0 + 2 * 0.938272))
    print(
        f"kinEnergy(1.0): momentum {s.momentum():.6f}, expected {p_gev:.6f} GeV/c for K = 1 GeV; "
        f"B_Rho {b.B_Rho():.6f} T m, expected {p_gev / 0.299792458:.6f} for p in GeV/c"
    )

    lat = TEAPOT_Lattice("ring")
    rf = RingRFTEAPOT("rf")
    rf.addRF(1, 1000.0, 0.0)  # 1000 "Volts" per the docstring, synchronous phase 0
    lat.addNode(DriftTEAPOT("d", 100.0))
    lat.addNode(rf)
    lat.initialize()
    C = lat.getLength()
    b.addParticle(0.0, 0.0, 0.0, 0.0, C / 4, 0.0)  # sin(2 pi z / C) = 1
    lat.trackBunch(b)
    # The kick is q V sin(h 2 pi z / C + phi_s) = q V; for q = 1 and V = 1 kV it is 1e-6 GeV.
    print(
        f"RF kick for V = 1000: observed {b.dE(0):.6e} GeV, expected {1e3 * 1e-9:.6e} GeV (1 kV, q = 1)"
    )
    print("DEFECT REPRODUCED" if abs(b.dE(0) - 1e-6) > 1e-9 else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

