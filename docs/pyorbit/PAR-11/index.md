---
title: "PAR-11: MAD and MAD-X solenoid import applies the bunch charge to KS a second time"
hide:
  - toc
---

# PAR-11 — MAD and MAD-X solenoid import applies the bunch charge to KS a second time

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Lattice import |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`teapot.py#L302-L308`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L302-L308)<br>[`teapot.py#L701-L716`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/teapot/teapot.py#L701-L716)<br>[`teapotbase.cc#L1367-L1375`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L1367-L1375) |

`readMADX` and `readMAD` copy a SOLENOID's strength KS into the parameter `B` of a TEAPOT solenoid, whose map multiplies `B` by the bunch charge number q. MAD defines KS = B_s/(Bρ), with B_s the longitudinal field and Bρ the beam particle's rigidity, so its map is the same for every charge. The imported solenoid applies q·KS instead: antiprotons, electrons and H⁻ ions (q = −1) rotate the opposite way, and a q = 2 ion by twice the angle. For KS = 0.5 m⁻¹ over 1 m and a particle starting at x = 1 mm, q = −1 leaves at y = +0.2397 mm and q = +2 at x = 0.7702 mm, where MAD-X gives y = −0.2397 mm and x = 0.9388 mm.

Run `python PAR-11.py`.

??? example "Reproducer: PAR-11.py"

    [Download PAR-11.py](PAR-11.py)

    ```python
    """PAR-11: MAD and MAD-X solenoid import applies the bunch charge to KS a second time

    `readMADX` and `readMAD` copy a SOLENOID's strength KS into the parameter `B` of a TEAPOT solenoid, whose map multiplies `B` by the bunch charge number q. MAD defines KS = B_s/(Bρ), with B_s the longitudinal field and Bρ the beam particle's rigidity, so its map is the same for every charge. The imported solenoid applies q·KS instead: antiprotons, electrons and H⁻ ions (q = −1) rotate the opposite way, and a q = 2 ion by twice the angle. For KS = 0.5 m⁻¹ over 1 m and a particle starting at x = 1 mm, q = −1 leaves at y = +0.2397 mm and q = +2 at x = 0.7702 mm, where MAD-X gives y = −0.2397 mm and x = 0.9388 mm.

    The script reads a 1 m MAD-X sequence and the equivalent MAD-8 line holding one solenoid with KS = 0.5 m⁻¹ and tracks a particle starting at x = 1 mm with zero energy deviation for q = +1, −1 (proton and electron mass), +2 and −2 at the same momentum. It compares the exit coordinates with the closed-form hard-edge solenoid map of normalized strength KS, which MAD-X applies for every charge. The q = +1 rows are the control, and each imported lattice is tracked again with its solenoid parameter set to KS/q, which gives the MAD map for every charge.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")
    import tempfile

    from orbit.core.bunch import Bunch
    from orbit.teapot import TEAPOT_Lattice

    KS, L = 0.5, 1.0
    X0 = (1e-3, 0.0, 0.0, 0.0)  # x, px, y, py


    def solenoid_map(ks, length, v):
        # Hard-edge solenoid of normalized strength ks = B_s/(B rho) at delta = 0, entrance and exit
        # fringes included (the MAD-X SOLENOID transfer matrix): K = ks/2, C = cos(K L), S = sin(K L).
        k = ks / 2.0
        c, s = math.cos(k * length), math.sin(k * length)
        x, px, y, py = v
        return (
            c * c * x + s * c / k * px + s * c * y + s * s / k * py,
            -k * s * c * x + c * c * px - k * s * s * y + s * c * py,
            -s * c * x - s * s / k * px + c * c * y + s * c / k * py,
            k * s * s * x - s * c * px - k * s * c * y + c * c * py,
        )


    # MAD: KS already carries the sign and size of the beam particle's charge, so the map is the same
    # for every q. For x0 = 1 mm: x = C^2 x0 = 0.9388 mm, y = -S C x0 = -0.2397 mm (K L = 0.25).
    EXPECTED = solenoid_map(KS, L, X0)

    SOURCES = {
        "readMADX": (
            "seq.madx",
            f"sol: solenoid, l={L}, ks={KS};\nring: sequence, l={L};\nsol, at={L / 2};\nendsequence;\n",
        ),
        "readMAD": ("seq.mad", f"sol: solenoid, l={L}, ks={KS}\nring: line=(sol)\n"),
    }
    SPECIES = [
        (+1.0, 0.93827208816, "proton, control"),
        (-1.0, 0.93827208816, "antiproton"),
        (-1.0, 0.00051099895, "electron"),
        (+2.0, 0.93827208816, "q = +2 ion"),
        (-2.0, 0.93827208816, "q = -2 ion"),
    ]


    def track(lat, q, m):
        b = Bunch()
        b.mass(m)
        b.charge(q)
        b.getSyncParticle().momentum(2.0)
        b.addParticle(X0[0], X0[1], X0[2], X0[3], 0.0, 0.0)
        lat.trackBunch(b)
        return (b.x(0), b.xp(0), b.y(0), b.yp(0))


    def fmt(v):
        return "(" + ", ".join(f"{u:+.6e}" for u in v) + ")"


    print(
        f"expected exit (x, px, y, py) {fmt(EXPECTED)} for every charge (MAD solenoid map, KS = {KS} m^-1, L = {L} m)"
    )
    bad, controls_ok = False, True
    for reader, (fname, text) in SOURCES.items():
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, fname)
            with open(path, "w") as f:
                f.write(text)
            lat = TEAPOT_Lattice()
            getattr(lat, reader)(path, "ring")
        sol = [n for n in lat.getNodes() if n.getType() == "solenoid teapot"]
        for q, m, label in SPECIES:
            for node in sol:
                node.setParam("B", KS)
            obs = track(lat, q, m)
            dev = max(abs(o - e) for o, e in zip(obs, EXPECTED))
            # Control: the parameter B that soln multiplies by q must be KS/q to give the MAD map.
            for node in sol:
                node.setParam("B", KS / q)
            dev_fix = max(abs(o - e) for o, e in zip(track(lat, q, m), EXPECTED))
            print(
                f"[{reader}, {label}, q = {q:+.0f}] B = {KS} as imported: exit observed {fmt(obs)}, max |diff| {dev:.3e}; "
                f"control with B = KS/q: max |diff| {dev_fix:.3e}"
            )
            controls_ok &= dev_fix < 1e-15
            if q == 1.0:
                controls_ok &= dev < 1e-15
            else:
                bad |= dev > 1e-6
    print("DEFECT REPRODUCED" if bad and controls_ok else "NOT REPRODUCED")

    ```

??? example "Reproducer: PAR-11-madx.py"

    [Download PAR-11-madx.py](PAR-11-madx.py)

    ```python
    """PAR-11: MAD-X gives the same solenoid and quadrupole maps for every beam charge

    Setup: needs cpymad (MAD-X 5), which the PyORBIT environment does not provide; run it with a Python that has cpymad.

    The script loads the 1 m solenoid of the main reproducer (KS = 0.5 m⁻¹) into MAD-X with BEAM particles proton, antiproton, electron and ions of charge ±2 at the same momentum, prints TWISS's transverse transfer matrix for each and tracks a particle from x = 1 mm with TRACK. All matrices and exit points are identical, so MAD-X's KS, like its K1, is normalized for the beam's own charge; a quadrupole with K1 = 0.5 m⁻² is the analogous control, whose charge-independent strength PyORBIT uses as given.
    """

    import numpy as np
    from cpymad.madx import Madx

    BEAMS = [
        "particle=proton",
        "particle=antiproton",
        "particle=electron",
        "particle=ion, mass=0.938272013, charge=2",
        "particle=ion, mass=0.938272013, charge=-2",
    ]
    for elem in ("sol: solenoid, l=1.0, ks=0.5;", "qf: quadrupole, l=1.0, k1=0.5;"):
        name = elem.split(":")[0]
        mats = []
        for beam in BEAMS:
            m = Madx(stdout=False)
            m.input(
                f"{elem}\nseq: sequence, l=1.0, refer=entry; e1: {name}, at=0; endsequence;\n"
                f"beam, {beam}, pc=2.0;\nuse, sequence=seq;"
            )
            tw = m.twiss(betx=1, bety=1, rmatrix=True)
            r = np.array([[tw[f"re{i}{j}"][-1] for j in range(1, 5)] for i in range(1, 5)])
            mats.append(r)
            print(
                f"MAD-X {name} BEAM {beam}: charge {m.sequence.seq.beam.charge:+.0f}, "
                f"R11 {r[0, 0]:+.9f} R13 {r[0, 2]:+.9f} R31 {r[2, 0]:+.9f} R33 {r[2, 2]:+.9f}"
            )
            m.quit()
        spread = max(np.max(np.abs(r - mats[0])) for r in mats)
        print(f"MAD-X {name}: max matrix difference between beam charges {spread:.3e}")

    # TRACK through the solenoid from x = 1 mm gives the same exit point for every charge too.
    for beam in BEAMS:
        m = Madx(stdout=False)
        m.input(
            "sol: solenoid, l=1.0, ks=0.5;\nseq: sequence, l=1.0, refer=entry; e1: sol, at=0; endsequence;\n"
            f"beam, {beam}, pc=2.0;\nuse, sequence=seq;"
        )
        m.input("track, onepass=true, onetable=true; start, x=1e-3; run, turns=1; endtrack;")
        t = m.table.trackone
        print(
            f"MAD-X TRACK sol BEAM {beam}: exit x {float(t.x[-1]):+.6e} px {float(t.px[-1]):+.6e} "
            f"y {float(t.y[-1]):+.6e} py {float(t.py[-1]):+.6e}"
        )
        m.quit()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

