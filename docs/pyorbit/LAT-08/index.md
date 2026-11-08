---
title: "LAT-08: BaseRF_Gap documents the matrix gap model as its default, but the default is RfGapTTF"
hide:
  - toc
---

# LAT-08 — `BaseRF_Gap` documents the matrix gap model as its default, but the default is `RfGapTTF`

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice API |
| Priority | P4 |
| Failure | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [LinacRfGapNodes.py#L40-L53](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L40-L53)<br>[LinacRfGapNodes.py#L82-L86](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L82-L86)<br>[LinacRfGapNodes.py#L112-L117](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L112-L117)<br>[LinacRfGapNodes.py#L243-L250](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L243-L250)<br>[sns_linac_lattice_factory.py#L215-L216](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/linac_parsers/sns_linac_lattice_factory.py#L215-L216) |

The `BaseRF_Gap` class docstring says the matrix RF gap model is used by default and lists E₀TL and the mode as its parameters. The constructor installs `RfGapTTF`, the transit-time-factor model, which uses E₀L and the four transit-time-factor polynomials and ignores E₀TL. A gap configured as the docstring describes, with only E₀TL set, therefore gives no acceleration, because E₀L defaults to zero. For a 2.5 MeV proton with E₀TL = 0.1200 MeV at a cavity phase of −20°, the energy gain is 0 instead of qE₀TL·cos φ = 0.1128 MeV. Lattices from the SNS lattice factory set both E₀TL and E₀L and are unaffected.

Run `python LAT-08.py`.

??? example "Reproducer: LAT-08.py"

    [Download LAT-08.py](LAT-08.py)

    ```python
    """LAT-08: `BaseRF_Gap` documents the matrix gap model as its default, but the default is `RfGapTTF`

    The `BaseRF_Gap` class docstring says the matrix RF gap model is used by default and lists E₀TL and the mode as its parameters. The constructor installs `RfGapTTF`, the transit-time-factor model, which uses E₀L and the four transit-time-factor polynomials and ignores E₀TL. A gap configured as the docstring describes, with only E₀TL set, therefore gives no acceleration, because E₀L defaults to zero. For a 2.5 MeV proton with E₀TL = 0.1200 MeV at a cavity phase of −20°, the energy gain is 0 instead of qE₀TL·cos φ = 0.1128 MeV. Lattices from the SNS lattice factory set both E₀TL and E₀L and are unaffected.

    The script builds a one-gap linac lattice, sets only E₀TL on the gap, runs the design pass of a 2.5 MeV proton and compares the reference energy gain with qE₀TL·cos φ, the gain of the documented matrix model. The controls install `MatrixRfGap` and `BaseRfGap` explicitly.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.core.linac import BaseRfGap, MatrixRfGap
    from orbit.py_linac.lattice import BaseRF_Gap, LinacAccLattice, RF_Cavity, Sequence

    E0TL = 1.2e-4  # GeV
    PHI = math.radians(-20.0)
    EKIN = 2.5e-3  # GeV
    # Thin matrix gap: the reference particle gains q*E0TL*cos(phi) (q = 1 for the default proton bunch).
    EXPECTED_MEV = E0TL * math.cos(PHI) * 1e3


    def design_gain(model):
        lat, seq, cav = LinacAccLattice("lat"), Sequence("seq"), RF_Cavity("cav")
        cav.setFrequency(402.5e6)
        cav.setPhase(PHI)
        gap = BaseRF_Gap("gap")
        # E0TL is the parameter the docstring lists for the default model; E0L stays 0.
        gap.setParam("E0TL", E0TL)
        if model is not None:
            gap.setCppGapModel(model)
        cav.addRF_GapNode(gap)
        seq.addNode(gap)
        seq.addRF_Cavity(cav)
        lat.addNode(gap)
        lat.initialize()
        b = Bunch()
        b.getSyncParticle().kinEnergy(EKIN)
        d = lat.trackDesignBunch(b)
        return type(gap.cppGapModel).__name__, (d.getSyncParticle().kinEnergy() - EKIN) * 1e3


    reproduced = False
    for label, model in (
        ("default model", None),
        ("control, explicit", MatrixRfGap()),
        ("control, explicit", BaseRfGap()),
    ):
        name, gain = design_gain(model)
        print(f"{label} {name}: gain {gain:.4f} MeV, expected q*E0TL*cos(phi) = {EXPECTED_MEV:.4f} MeV")
        if model is None:
            reproduced = abs(gain - EXPECTED_MEV) > 1e-9
        elif abs(gain - EXPECTED_MEV) > 1e-9:
            raise RuntimeError(f"control case failed: {name} does not give q*E0TL*cos(phi)")
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

