---
title: "IO-09: Changing the bunch mass leaves the synchronous kinetic energy, β and γ stale"
hide:
  - toc
---

# IO-09 — Changing the bunch mass leaves the synchronous kinetic energy, β and γ stale

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Particles and I/O |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [Bunch.cc#L789-L794](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Bunch.cc#L789-L794)<br>[SyncPart.cc#L236-L306](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SyncPart.cc#L236-L306)<br>[SyncPart.cc#L133-L164](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SyncPart.cc#L133-L164)<br>[SyncPart.cc#L97-L99](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SyncPart.cc#L97-L99)<br>[SyncPart.cc#L193-L203](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/SyncPart.cc#L193-L203)<br>[teapotbase.cc#L158-L184](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/teapot/teapotbase.cc#L158-L184) |

`Bunch.mass(value)` updates the mass and classical radius but not the synchronous
particle's cached kinetic energy, β and γ, which are recomputed only when its momentum or kinetic
energy is set. The reference keeps its momentum, but `kinEnergy()`, `beta()`, `gamma()` and the maps
that read them keep the old species' values. A default proton bunch set to 2.5 MeV and then given
the H⁻ mass 0.939294 GeV keeps K = 2.500 MeV and β₀ = 0.07285 instead of the 2.497 MeV and 0.07278
its momentum implies, β₀ 0.1083% high. Changed to the electron mass, a 1 GeV proton reference keeps
β₀ = 0.8750 instead of 1.000, and a 1 m TEAPOT drift shifts z of a particle with dE = 1 MeV by
1.577×10⁻⁴ m instead of 5.347×10⁻¹¹ m. Setting the mass first gives consistent values.

Run `python IO-09.py`.

??? example "Reproducer: IO-09.py"

    [Download IO-09.py](IO-09.py)

    ```python
    """IO-09: Changing the bunch mass leaves the synchronous kinetic energy, β and γ stale

    `Bunch.mass(value)` updates the mass and classical radius but not the synchronous particle's cached
    kinetic energy, β and γ, which are recomputed only when its momentum or kinetic energy is set. The
    reference keeps its momentum, but `kinEnergy()`, `beta()`, `gamma()` and the maps that read them
    keep the old species' values. A default proton bunch set to 2.5 MeV and then given the H⁻ mass
    0.939294 GeV keeps K = 2.500 MeV and β₀ = 0.07285 instead of the 2.497 MeV and 0.07278 its momentum
    implies, β₀ 0.1083% high. Changed to the electron mass, a 1 GeV proton reference keeps β₀ = 0.8750
    instead of 1.000, and a 1 m TEAPOT drift shifts z of a particle with dE = 1 MeV by 1.577×10⁻⁴ m
    instead of 5.347×10⁻¹¹ m. Setting the mass first gives consistent values.

    The script sets a proton reference to 1 GeV, changes the mass to the electron mass and compares the
    reference's K, β and γ with the relativistic relations for the momentum it retains. It then tracks a
    particle with dE = 1 MeV through a 1 m `DriftTEAPOT` and compares z with TEAPOT's drift formula
    evaluated with the consistent β₀ and γ₀. A second case repeats the comparison for a default bunch at
    2.5 MeV given the H⁻ mass afterwards; the control sets the electron mass first and the same momentum
    afterwards.
    """

    import math

    from orbit.core.bunch import Bunch
    from orbit.teapot import DriftTEAPOT, TEAPOT_Lattice

    M_P, M_E = 0.93827208816, 0.00051099895  # GeV, CODATA 2018
    L, DE = 1.0, 1.0e-3  # drift length [m], particle energy deviation [GeV]


    def drift_z(b):
        lat = TEAPOT_Lattice("drift")
        lat.addNode(DriftTEAPOT("d", L))
        lat.initialize()
        b.addParticle(0.0, 0.0, 0.0, 0.0, 0.0, DE)
        lat.trackBunch(b)
        return b.z(0)


    def expected(p, m):
        """K, beta, gamma and the drift's z for momentum p and mass m (GeV)."""
        e = math.sqrt(p * p + m * m)
        beta, gamma = p / e, e / m
        # TEAPOT drift (teapotbase.cc drift): dp_p = dE/(p0 beta0), KNL = 1/(1 + dp_p),
        # z -= L * ((dp_p^2/gamma0^2)/2 * KNL - dp_p/gamma0^2) * KNL for xp = yp = 0.
        dp_p = DE / (p * beta)
        knl = 1.0 / (1.0 + dp_p)
        z = -L * ((dp_p * dp_p / gamma**2) / 2.0 * knl - dp_p / gamma**2) * knl
        return e - m, beta, gamma, z


    def report(label, b):
        sp = b.getSyncParticle()
        p, m = sp.momentum(), b.mass()
        obs = (sp.kinEnergy(), sp.beta(), sp.gamma())
        exp = expected(p, m)
        z = drift_z(b)
        print(
            f"{label}: p = {p:.9f} GeV/c, m = {m:.9f} GeV; K observed {obs[0]:.9f}, expected {exp[0]:.9f} "
            f"(sqrt(p^2+m^2)-m); beta observed {obs[1]:.9f}, expected {exp[1]:.9f} (p/sqrt(p^2+m^2)); "
            f"gamma observed {obs[2]:.6f}, expected {exp[2]:.6f} (sqrt(p^2+m^2)/m)"
        )
        print(
            f"{label}: drift z observed {z:.4e} m, expected {exp[3]:.4e} m (TEAPOT drift with consistent beta0, gamma0)"
        )
        print(
            f"{label}: relative error K {obs[0] / exp[0] - 1:.4e}, beta {obs[1] / exp[1] - 1:.4e}, gamma {obs[2] / exp[2] - 1:.4e}"
        )
        pairs = [*zip(obs, exp[:3]), (z, exp[3])]
        return all(abs(o - e) <= 1e-12 * abs(e) for o, e in pairs)


    stale = Bunch()
    stale.mass(M_P)
    stale.getSyncParticle().kinEnergy(1.0)
    stale.mass(M_E)
    p_kept = stale.getSyncParticle().momentum()
    consistent_after = report("proton at 1 GeV, then mass set to the electron mass", stale)

    # A default (proton) bunch set to 2.5 MeV, then given the H- mass, as the template bunch of the
    # upstream example examples/SNS_Linac/linac_diagnostic_nodes/linac_bpm_node_test.py is.
    hminus = Bunch()
    hminus.getSyncParticle().kinEnergy(0.0025)
    hminus.mass(0.939294)
    consistent_hminus = report("default proton bunch at 2.5 MeV, then the H- mass 0.939294 GeV", hminus)

    ctl = Bunch()
    ctl.mass(M_E)
    ctl.getSyncParticle().momentum(p_kept)
    consistent_ctl = report("electron mass set first, same momentum (control)", ctl)

    print(
        "DEFECT REPRODUCED"
        if not consistent_after and not consistent_hminus and consistent_ctl
        else "NOT REPRODUCED"
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

