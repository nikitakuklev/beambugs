---
title: "TRK-02: MAD RFCAVITY import uses VOLT in MV as GV and reverses the sign of the energy kick"
hide:
  - toc
---

# TRK-02 — MAD RFCAVITY import uses VOLT in MV as GV and reverses the sign of the energy kick

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot.py#L335-L351`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L335-L351)<br>[`teapotbase.cc#L1527-L1567`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L1527-L1567) |

The TEAPOT factory behind `readMAD` turns an RFCAVITY into `RingRFTEAPOT` with the voltage unchanged and the phase φs = −2π·LAG; the node adds q·V·sin(2πhz/C + φs) GeV to dE, with h the harmonic number, C the ring length and z the distance ahead of the reference. VOLT is in MV, so it acts 1000 times too strong, and MAD's kick is VOLT·10⁻³·sin(2π·LAG − 2πhz/C) GeV, so the imported kick also has the wrong sign; the correct conversion is V = VOLT/1000 and φs = π − 2π·LAG. In a 10 m line, VOLT = 0.01 and LAG = 0.25 change the reference energy by −1.000×10⁻² GeV instead of +1.000×10⁻⁵ GeV. Below transition the sign alone makes a stable bucket unstable.

Run `python TRK-02.py`.

??? example "Reproducer: TRK-02.py"

    [Download TRK-02.py](TRK-02.py)

    ```python
    """TRK-02: MAD RFCAVITY import uses VOLT in MV as GV and reverses the sign of the energy kick

    The TEAPOT factory behind `readMAD` turns an RFCAVITY into `RingRFTEAPOT` with the voltage unchanged and the phase φs =
    −2π·LAG; the node adds q·V·sin(2πhz/C + φs) GeV to dE, with h the harmonic number, C the ring length and z the distance
    ahead of the reference. VOLT is in MV, so it acts 1000 times too strong, and MAD's kick is VOLT·10⁻³·sin(2π·LAG −
    2πhz/C) GeV, so the imported kick also has the wrong sign; the correct conversion is V = VOLT/1000 and φs = π − 2π·LAG.
    In a 10 m line, VOLT = 0.01 and LAG = 0.25 change the reference energy by −1.000×10⁻² GeV instead of +1.000×10⁻⁵ GeV.
    Below transition the sign alone makes a stable bucket unstable.

    The script writes a cavity into a temporary MAD-8 line, reads it with `readMAD`, prints the stored voltage and phase, and
    tracks one particle per case: at LAG = 0.25 the reference energy change, at LAG = 0 the dE of a particle 0.5 m ahead,
    both against MAD's kick VOLT·10⁻³·sin(2π·LAG − 2πhz/C) GeV. It then tracks a particle starting at z = 1 mm for 400 turns
    in a drift-only ring, which is below transition, through a 0.5 MV cavity at LAG = 0 with the factory's phase −2π·LAG.
    The control is a hand-built `RingRFTEAPOT` with V = VOLT/1000 and φs = π − 2π·LAG, which reproduces MAD's kick and keeps
    the particle in the bucket.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import DriftTEAPOT, RingRFTEAPOT, TEAPOT_Lattice

    C, H = 10.0, 1  # ring length (m), harmonic number


    def read_mad(volt_mv, lag):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "line.mad")
            with open(path, "w") as f:
                f.write(
                    f"rf: rfcavity, l=1.0, volt={volt_mv}, lag={lag}, harmon={H}\nd: drift, l=9.0\nring: line=(d, rf)\n"
                )
            lat = TEAPOT_Lattice()
            lat.readMAD(path, "ring")
        return lat


    def hand_built(volt_mv, phase_s):
        """The imported layout with a cavity of V = VOLT/1000 GV and the given phase."""
        lat = TEAPOT_Lattice("ring")
        rf = RingRFTEAPOT("rf")
        rf.addRF(H, volt_mv * 1e-3, phase_s)
        lat.addNode(DriftTEAPOT("d1", 9.5))
        lat.addNode(rf)
        lat.addNode(DriftTEAPOT("d2", 0.5))
        lat.initialize()
        return lat


    def bunch(z):
        b = Bunch()
        b.mass(0.938272)
        b.charge(1)
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(0.0, 0.0, 0.0, 0.0, z, 0.0)
        return b


    def correct_phase(lag):
        # V*sin(2*pi*h*z/C + pi - 2*pi*LAG) = V*sin(2*pi*LAG - 2*pi*h*z/C), MAD's kick
        return math.pi - 2 * math.pi * lag


    def mad_kick(volt_mv, lag, z):
        # MAD-8 and MAD-X: dE = VOLT[MV]*1e-3 GeV * sin(2*pi*LAG - omega*T/c), T = -c*dt = z/beta and
        # omega = 2*pi*h*beta*c/C, so omega*T/c = 2*pi*h*z/C (z > 0 ahead of the reference).
        return volt_mv * 1e-3 * math.sin(2 * math.pi * lag - 2 * math.pi * H * z / C)


    def one_pass(lat, lag, z):
        b = bunch(z)
        k0 = b.getSyncParticle().kinEnergy()
        lat.trackBunch(b)
        # LAG = 0.25 case: the reference energy change. LAG = 0 case: phi_s = 0 leaves the reference unchanged, so the
        # particle's dE is its whole kick (the reference gain is not also added to dE, unlike at phi_s != 0).
        return b.getSyncParticle().kinEnergy() - k0 if z == 0.0 else b.dE(0)


    def max_abs_z(lat, turns=400):
        b = bunch(1e-3)
        zmax = 0.0
        for _ in range(turns):
            lat.trackBunch(b)
            zmax = max(zmax, abs(b.z(0)))
        return zmax


    reproduced = True
    for lag, z, what in ((0.25, 0.0, "reference energy change"), (0.0, 0.5, "particle dE")):
        lat = read_mad(0.01, lag)
        rf = next(n for n in lat.getNodes() if isinstance(n, RingRFTEAPOT))
        obs, ctl, exp = (
            one_pass(lat, lag, z),
            one_pass(hand_built(0.01, correct_phase(lag)), lag, z),
            mad_kick(0.01, lag, z),
        )
        print(
            f"[VOLT=0.01 MV, LAG={lag}, z={z} m] readMAD stores V {rf.getParam('voltages')}, phase {rf.getParam('phases')}; "
            f"{what} observed {obs:.6e} GeV, expected {exp:.6e} GeV (MAD kick); control V/1000, pi - 2 pi LAG: {ctl:.6e} GeV"
        )
        reproduced &= abs(obs - exp) > 1e-3 * abs(exp) and abs(ctl - exp) < 1e-9 * abs(exp)

    # Stability below transition (a drift-only ring has zero momentum compaction): MAD's LAG = 0 kick -sin(2*pi*h*z/C)
    # decelerates a particle ahead, which then falls back, so |z| oscillates with its starting amplitude of 1 mm. The sign case
    # keeps the voltage right (0.5 MV as 5e-4 GV) and uses the factory's phase -2*pi*LAG = 0, isolating the sign reversal.
    z_sign = max_abs_z(hand_built(0.5, -2 * math.pi * 0.0))
    z_ctl = max_abs_z(hand_built(0.5, correct_phase(0.0)))
    print(
        f"[VOLT=0.5 MV, LAG=0, drift-only ring, 400 turns] factory phase -2 pi LAG: max |z| observed {z_sign:.4e} m, "
        f"expected 1.0000e-03 m (stable bucket below transition); control pi - 2 pi LAG: {z_ctl:.4e} m"
    )
    reproduced &= z_sign > 1e-2 and abs(z_ctl - 1e-3) < 1e-6
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

??? example "Reproducer: TRK-02-madx.py"

    [Download TRK-02-madx.py](TRK-02-madx.py)

    ```python
    """TRK-02: MAD-X tracking of the same RFCAVITY, the reference for the kick formula

    Setup: needs cpymad (MAD-X), which a PyORBIT environment usually lacks, so it runs in a separate interpreter; it does
    not import PyORBIT.

    The script tracks one proton at 1 GeV kinetic energy for one turn of a 10 m sequence holding a 1 m RFCAVITY with
    VOLT = 0.01 MV and HARMON = 1, at three (LAG, T) pairs, and prints the energy change PT·pc next to
    VOLT·10⁻³·sin(2π·LAG − 2πz/C) GeV with z = β·T, the formula the main script uses as its expected value.
    """

    import math

    from cpymad.madx import Madx

    K, M, C = 1.0, 0.938272088, 10.0  # GeV, GeV, m
    E = K + M
    P = math.sqrt(E * E - M * M)
    BETA = P / E

    agree = True
    for lag, t in ((0.25, 0.0), (0.0, 0.5), (0.1, -0.3)):
        mx = Madx(stdout=False)
        mx.input(f"""
        beam, particle=proton, energy={E};
        rf: rfcavity, l=1.0, volt=0.01, lag={lag}, harmon=1;
        d: drift, l=9.0;
        ring: sequence, l={C}; d, at=4.5; rf, at=9.5; endsequence;
        use, sequence=ring;
        track, onepass, onetable;
        start, x=0, px=0, y=0, py=0, t={t}, pt=0;
        run, turns=1;
        endtrack;
        """)
        de = mx.table.trackone.pt[-1] * P  # PT = dE/(p0 c)
        mx.quit()
        # MAD-X T = -c*dt is positive ahead; PyORBIT z = -beta*c*dt, so z = beta*T
        z = BETA * t
        formula = 0.01e-3 * math.sin(2 * math.pi * lag - 2 * math.pi * z / C)
        print(
            f"[LAG={lag}, T={t} m, z={z:.9f} m] MAD-X dE {de:.9e} GeV, formula VOLT*1e-3*sin(2 pi LAG - 2 pi z/C) {formula:.9e} GeV"
        )
        agree &= abs(de - formula) < 1e-6 * abs(formula)
    print("MAD-X AGREES WITH THE FORMULA" if agree else "MAD-X DIFFERS FROM THE FORMULA")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

