---
title: "OPT-03: getRingParametersDict reports no tunes for a stable coupled ring when one uncoupled block is unstable"
hide:
  - toc
---

# OPT-03 — `getRingParametersDict` reports no tunes for a stable coupled ring when one uncoupled block is unstable

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Optics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`MATRIX_Lattice.py#L130-L159`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L130-L159) |

`MATRIX_Lattice.getRingParametersDict` judges stability from the two uncoupled 2×2 diagonal blocks of the transverse one-turn matrix and, when either has |cos μ| ≥ 1 (μ the phase advance), returns None for every tune, Twiss and dispersion entry, the coupled eigentunes included. With transverse coupling a diagonal block can have |cos μ| > 1 while all four eigenvalues of the 4×4 one-turn matrix lie on the unit circle, so the motion is stable and its eigentunes are defined. A four-cell FODO ring with a 1 m solenoid of strength kₛ = Bₛ/(Bρ) = 0.5 m⁻¹, Bρ the magnetic rigidity, has eigentunes 0.1530 and 0.4368 and x-block cos μ = −1.073; the method returns None for both, and `getTunes` raises TypeError.

Run `python OPT-03.py`.

??? example "Reproducer: OPT-03.py"

    [Download OPT-03.py](OPT-03.py)

    ```python
    """OPT-03: `getRingParametersDict` reports no tunes for a stable coupled ring when one uncoupled block is unstable

    `MATRIX_Lattice.getRingParametersDict` judges stability from the two uncoupled 2×2 diagonal blocks
    of the transverse one-turn matrix and, when either has |cos μ| ≥ 1 (μ the phase advance), returns
    None for every tune, Twiss and dispersion entry, the coupled eigentunes included. With transverse
    coupling a diagonal block can have |cos μ| > 1 while all four eigenvalues of the 4×4 one-turn matrix
    lie on the unit circle, so the motion is stable and its eigentunes are defined. A four-cell FODO
    ring with a 1 m solenoid of strength kₛ = Bₛ/(Bρ) = 0.5 m⁻¹, Bρ the magnetic rigidity, has
    eigentunes 0.1530 and 0.4368 and x-block cos μ = −1.073; the method returns None for both, and
    `getTunes` raises TypeError.

    The script builds the ring with the solenoid and, as a control, with a 1 m drift in its place. It
    prints the method's eigentunes and plane tunes, the moduli of the eigenvalues of the 4×4 one-turn
    matrix and the cos μ of its diagonal blocks, and calls `getTunes`. The expected eigentunes are
    arccos(Re λ)/2π of the eigenvalues λ, the formula the method applies when it does not return early.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.teapot import (
        DriftTEAPOT,
        QuadTEAPOT,
        SolenoidTEAPOT,
        TEAPOT_Lattice,
        TEAPOT_MATRIX_Lattice,
    )


    def make_ring(solenoid):
        lat = TEAPOT_Lattice("ring")
        for c in range(4):
            lat.addNode(QuadTEAPOT(f"qf{c}", 0.25, 2, 0.95))
            lat.addNode(DriftTEAPOT(f"a{c}", 2.0))
            lat.addNode(QuadTEAPOT(f"qd{c}", 0.5, 2, -1.65))
            lat.addNode(DriftTEAPOT(f"b{c}", 2.0))
            lat.addNode(QuadTEAPOT(f"qe{c}", 0.25, 2, 0.95))
        if solenoid:
            # SolenoidTEAPOT(name, B, length, nparts): B is ks = Bs/(B rho) in 1/m; the kernel's map
            # equals the MAD solenoid matrix with Larmor angle ks*L/2
            lat.addNode(SolenoidTEAPOT("sol", 0.5, 1.0, 1))
        else:
            lat.addNode(DriftTEAPOT("sol", 1.0))
        lat.initialize()
        return lat


    bad = False
    for solenoid in (False, True):
        ref = Bunch()
        ref.mass(0.938272)
        ref.charge(1)
        ref.getSyncParticle().kinEnergy(1.0)  # GeV
        ml = TEAPOT_MATRIX_Lattice(make_ring(solenoid), ref)
        d = ml.getRingParametersDict()
        mt = ml.getOneTurnMatrix()
        M = np.array([[mt.get(i, j) for j in range(4)] for i in range(4)])
        ev = np.linalg.eigvals(M)
        # Stable linear motion: all four |lambda| = 1. One eigentune per conjugate pair,
        # arccos(Re lambda)/2pi.
        exp = sorted({round(float(np.arccos(np.clip(lam.real, -1, 1)) / (2 * np.pi)), 9) for lam in ev})
        obs = [d["fractional tune 1"], d["fractional tune 2"]]
        obs_txt = "None, None" if obs[0] is None else f"{min(obs):.6f}, {max(obs):.6f}"
        nux, nuy = d["fractional tune x"], d["fractional tune y"]
        plane_txt = "None, None" if nux is None else f"{nux:.6f}, {nuy:.6f}"
        tag = "solenoid" if solenoid else "control, drift"
        print(
            f"[{tag}] max ||lambda| - 1| = {np.max(np.abs(np.abs(ev) - 1)):.1e}; "
            f"diagonal-block cos mu: x {(M[0, 0] + M[1, 1]) / 2:.6f}, y {(M[2, 2] + M[3, 3]) / 2:.6f}"
        )
        print(
            f"  eigentunes: observed {obs_txt}, expected {exp[0]:.6f}, {exp[1]:.6f} "
            f"(arccos(Re lambda)/2pi); plane tunes observed {plane_txt}"
        )
        try:
            t = ml.getTunes()
            print(f"  getTunes: {t[0]:.6f}, {t[1]:.6f}")
        except TypeError as e:
            print(f"  getTunes: TypeError: {e}")
        stable = np.max(np.abs(np.abs(ev) - 1.0)) < 1e-9
        if solenoid and stable and obs[0] is None:
            bad = True
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small class="bb-scope">source reading</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source reading |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

