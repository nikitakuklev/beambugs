---
title: "TRK-16: The linac-tracker quadrupole drops the path lengthening of transverse slopes, so it does not reduce to the linac drift"
hide:
  - toc
---

# TRK-16 — The linac-tracker quadrupole drops the path lengthening of transverse slopes, so it does not reduce to the linac drift

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P4 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`linac_tracking.cc#L126-L234`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/tracking/linac_tracking.cc#L126-L234)<br>[`linac_tracking.cc#L96-L103`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/tracking/linac_tracking.cc#L96-L103)<br>[`LinacAccNodes.py#L63-L70`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L63-L70)<br>[`LinacAccNodes.py#L457-L474`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacAccNodes.py#L457-L474) |

With `setLinacTracker(True)`, linac quadrupoles call `linac_quad1`, which advances z by (β − β₀)·L/β₀ with β the particle's total speed, β₀ the reference speed and L the step length. The longitudinal speed belongs there: a particle moving at slope θ = pₓ/p₀ travels a longer path and falls behind by about Lθ²/2, as the same tracker's drift and the TEAPOT quadrupole both give. The linac quadrupole therefore does not approach the drift as its gradient goes to zero. For 0.1 GeV protons through a 0.1 m quadrupole at θ = 3 mrad, z stays 0 at every gradient instead of −0.4500 µm at a vanishing gradient and −0.4353 µm at kq = 10 m⁻².

Run `python TRK-16.py`.

??? example "Reproducer: TRK-16.py"

    [Download TRK-16.py](TRK-16.py)

    ```python
    """TRK-16: The linac-tracker quadrupole drops the path lengthening of transverse slopes, so it does not reduce to the linac drift

    With `setLinacTracker(True)`, linac quadrupoles call `linac_quad1`, which advances z by (β − β₀)·L/β₀ with β the particle's total speed, β₀ the reference speed and L the step length. The longitudinal speed belongs there: a particle moving at slope θ = pₓ/p₀ travels a longer path and falls behind by about Lθ²/2, as the same tracker's drift and the TEAPOT quadrupole both give. The linac quadrupole therefore does not approach the drift as its gradient goes to zero. For 0.1 GeV protons through a 0.1 m quadrupole at θ = 3 mrad, z stays 0 at every gradient instead of −0.4500 µm at a vanishing gradient and −0.4353 µm at kq = 10 m⁻².

    The script tracks one reference-energy proton with slope 3 mrad through `linac_tracking.quad1` at strengths from 10⁻⁹ to 10 m⁻² and compares z with −½∫θ(s)² ds over the quadrupole's own linear solution θ(s). It repeats the zero-gradient limit through a 0.1 m linac `Quad` with dB/dr = 10⁻⁹ T/m in a `LinacAccLattice` with the linac tracker on. The linac drift of the same length, exact here as L(√(1 − θ²) − 1), is the control.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    from orbit.core.bunch import Bunch
    from orbit.core.linac import linac_tracking as LT
    from orbit.py_linac.lattice import Drift, LinacAccLattice, Quad, Sequence

    K0, L, XP = 0.1, 0.1, 3.0e-3  # GeV, m, rad


    def bunch():
        b = Bunch()
        b.mass(0.938272013)
        b.charge(1.0)
        b.getSyncParticle().kinEnergy(K0)
        b.addParticle(0.0, XP, 0.0, 0.0, 0.0, 0.0)
        return b


    def kernel(kq):
        b = bunch()
        if kq is None:
            LT.drift(b, L)
        else:
            LT.quad1(b, L, kq)
        return b.z(0)


    def lattice(node):
        lat = LinacAccLattice("l")
        seq = Sequence("s")
        seq.setLinacAccLattice(lat)
        node.setLength(L)
        seq.addNode(node)
        lat.addNode(node)
        lat.initialize()
        lat.setLinacTracker(True)
        b = bunch()
        lat.trackBunch(b)
        return b.z(0)


    defects, controls_ok = 0, True
    # Control: at the reference energy the linac drift advances z by L (p_z/p0 - 1) = L (sqrt(1 - xp^2) - 1).
    z_drift = L * (math.sqrt(1.0 - XP * XP) - 1.0)
    zd = kernel(None)
    print(
        f"[linac_tracking.drift (control)] z observed {zd:.6e} m, expected L(sqrt(1 - xp^2) - 1) = {z_drift:.6e} m"
    )
    controls_ok &= abs(zd - z_drift) < 1e-15
    for kq in (1e-9, 1e-3, 1.0, 10.0):
        s = math.sqrt(kq)
        # Linear focusing solution x' = XP cos(s u), so int_0^L x'^2 du = XP^2 (L/2 + sin(2 s L)/(4 s)) and the
        # longer path delays the particle by z = -1/2 of that integral.
        expected = -0.5 * XP * XP * (L / 2 + math.sin(2 * s * L) / (4 * s))
        z = kernel(kq)
        print(
            f"[linac_tracking.quad1, kq = {kq:g} m^-2] z observed {z:.6e} m, expected -1/2 int x'^2 ds = {expected:.6e} m"
        )
        defects += abs(z - expected) > 1e-3 * abs(expected)

    z_lat_drift = lattice(Drift("d"))
    q = Quad("q")
    q.setParam("dB/dr", 1e-9)
    z_lat_quad = lattice(q)
    print(
        f"[LinacAccLattice, linac tracker, Drift (control)] z observed {z_lat_drift:.6e} m, expected {z_drift:.6e} m"
    )
    print(
        f"[LinacAccLattice, linac tracker, Quad dB/dr = 1e-9 T/m] z observed {z_lat_quad:.6e} m, expected {z_drift:.6e} m "
        f"(the drift, the zero-gradient limit)"
    )
    controls_ok &= abs(z_lat_drift - z_drift) < 1e-15
    defects += abs(z_lat_quad - z_drift) > 1e-3 * abs(z_drift)

    print("DEFECT REPRODUCED" if defects == 5 and controls_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

