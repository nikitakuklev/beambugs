---
title: "PAR-02: readMADX turns every RFCAVITY into a drift"
hide:
  - toc
---

# PAR-02 — `readMADX` turns every RFCAVITY into a drift

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">known-limitation</span> [?](../../methodology.md#tags) |
| Source | [`madx_parser.py#L760-L775`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/parsers/madx_parser.py#L760-L775)<br>[`teapot.py#L335-L351`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L335-L351) |

`TEAPOT_Lattice.readMADX` changes the type of every RFCAVITY to a drift before the TEAPOT element factory sees it, so the imported lattice has no RF node: VOLT, LAG and HARMON are dropped with no message, and the ring has no longitudinal focusing or acceleration. MAD-X gives a particle crossing an RFCAVITY at the reference time the energy kick VOLT·sin(2π·LAG), with VOLT in MV. For `rf: rfcavity, l=1, volt=0.01, lag=0.25, harmon=1` an on-axis proton gains 0 instead of 1.000×10⁻⁵ GeV, and the cavity is a plain 1 m drift.

Run `python PAR-02.py`.

??? example "Reproducer: PAR-02.py"

    [Download PAR-02.py](PAR-02.py)

    ```python
    """PAR-02: `readMADX` turns every RFCAVITY into a drift

    `TEAPOT_Lattice.readMADX` changes the type of every RFCAVITY to a drift before the TEAPOT element factory sees it, so the imported lattice has no RF node: VOLT, LAG and HARMON are dropped with no message, and the ring has no longitudinal focusing or acceleration. MAD-X gives a particle crossing an RFCAVITY at the reference time the energy kick VOLT·sin(2π·LAG), with VOLT in MV. For `rf: rfcavity, l=1, volt=0.01, lag=0.25, harmon=1` an on-axis proton gains 0 instead of 1.000×10⁻⁵ GeV, and the cavity is a plain 1 m drift.

    The script writes a 10 m MAD-X sequence holding the cavity to a temporary directory, reads it with `readMADX`, prints the cavity node's type and the number of `RingRFTEAPOT` nodes, and tracks an on-axis proton through one pass. The expected gain is the MAD-X RFCAVITY kick at the reference time. The control reads the same cavity from a MAD-8 file with `readMAD`, where the factory's RFCAVITY branch builds one `RingRFTEAPOT` node.
    """

    import contextlib
    import io
    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import RingRFTEAPOT, TEAPOT_Lattice


    def read(text, reader, name):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, name)
            with open(path, "w") as f:
                f.write(text)
            lat = TEAPOT_Lattice()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                getattr(lat, reader)(path, "ring")
        return lat, out.getvalue()


    lat, printed = read(
        "rf: rfcavity, l=1, volt=0.01, lag=0.25, harmon=1;\nring: sequence, l=10;\nrf, at=2;\nendsequence;\n",
        "readMADX",
        "seq.madx",
    )
    rf_types = [n.getType() for n in lat.getNodes() if n.getName() == "rf"]
    n_rf = sum(1 for n in lat.getNodes() if isinstance(n, RingRFTEAPOT))
    print(
        f"[readMADX] cavity node types {rf_types}, RingRFTEAPOT nodes {n_rf}, expected >= 1; reader messages: {bool(printed.strip())}"
    )

    b = Bunch()
    b.mass(0.938272)
    b.charge(1)
    b.getSyncParticle().kinEnergy(1.0)
    b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K0 = b.getSyncParticle().kinEnergy()
    lat.trackBunch(b)
    gain = (
        b.getSyncParticle().kinEnergy() - K0 + b.dE(0)
    )  # reference energy change plus the particle's dE
    # MAD-X RFCAVITY: dE = VOLT * sin(2*pi*(LAG - h*f0*t)); at t = 0, VOLT = 0.01 MV = 1e-5 GeV, LAG = 0.25 gives sin = 1.
    expected = 0.01e-3 * math.sin(2 * math.pi * 0.25)
    print(
        f"[readMADX] on-axis energy gain observed {gain:.6e} GeV, expected {expected:.6e} GeV (VOLT*sin(2*pi*LAG))"
    )

    lat8, _ = read(
        "rf: rfcavity, l=1.0, volt=0.01, lag=0.25, harmon=1\nd: drift, l=4.5\nring: line=(d, rf, d)\n",
        "readMAD",
        "line.mad",
    )
    n_rf8 = sum(1 for n in lat8.getNodes() if isinstance(n, RingRFTEAPOT))
    print(f"[control, readMAD] RingRFTEAPOT nodes {n_rf8}, expected 1")
    print("DEFECT REPRODUCED" if n_rf == 0 and abs(gain - expected) > 1e-9 else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-02-madx.py"

    [Download PAR-02-madx.py](PAR-02-madx.py)

    ```python
    """PAR-02: MAD-X tracks the same cavity with an energy gain of VOLT·sin(2π·LAG)

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script loads the sequence of the main reproducer into MAD-X, tracks one on-axis proton at the reference time through one pass and prints its energy change pt·p₀c. MAD-X gives 1.000×10⁻⁵ GeV, VOLT·sin(2π·LAG) with VOLT = 0.01 MV.
    """

    import os
    import tempfile

    from cpymad.madx import Madx

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "seq.madx")
        with open(path, "w") as f:
            f.write(
                "rf: rfcavity, l=1, volt=0.01, lag=0.25, harmon=1;\nring: sequence, l=10;\nrf, at=2;\nendsequence;\n"
            )
        m = Madx(stdout=False)
        m.input("beam, particle=proton, energy=1.938272;")  # total energy: kinetic 1 GeV
        m.call(path)
        m.input("use, sequence=ring;")
        m.input("track, onepass=true, onetable=true; start; run, turns=1; endtrack;")
        pc = m.sequence.ring.beam.pc
        # MAD-X pt = dE/(p0 c), so the energy change is pt * p0c in GeV.
        dE = float(m.table.trackone.pt[-1]) * pc
        print(f"MAD-X energy change {dE:.6e} GeV, VOLT*sin(2*pi*LAG) = 1.000000e-05 GeV")
        m.quit()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

