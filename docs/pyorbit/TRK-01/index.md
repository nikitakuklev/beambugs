---
title: "TRK-01: The TTF RF gap kernels, default RfGapTTF included, apply the particle phase correction with the wrong sign"
hide:
  - toc
---

# TRK-01 — The TTF RF gap kernels, default `RfGapTTF` included, apply the particle phase correction with the wrong sign

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Tracking |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`RfGapTTF.cc#L71-L74`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapTTF.cc#L71-L74)<br>[`RfGapTTF.cc#L104-L107`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapTTF.cc#L104-L107)<br>[`RfGapTTF_slow.cc#L161-L163`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapTTF_slow.cc#L161-L163)<br>[`RfGapThreePointTTF.cc#L111-L114`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapThreePointTTF.cc#L111-L114)<br>[`RfGapThreePointTTF_slow.cc#L130-L132`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/linac/rfgap/RfGapThreePointTTF_slow.cc#L130-L132)<br>[`LinacRfGapNodes.py#L82-L86`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L82-L86)<br>[`LinacRfGapNodes.py#L480-L491`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/py_linac/lattice/LinacRfGapNodes.py#L480-L491) |

The default thin RF gap `RfGapTTF` delays the reference by Δφ₀ = qE₀Lκ(T′ sin φ + S′ cos φ)/(mβ²γ³), with E₀L the gap's field integral, φ its RF phase, κ = 2πf/(βc) and T′, S′ the κ-derivatives of the transit-time factors, but adds each particle's own correction Δφ to κz instead of subtracting it. A particle delayed more than the reference falls back, so κz must decrease by Δφ − Δφ₀, as direct integration through the field confirms. For a 2.5 MeV proton at 402.5 MHz, E₀L = 0.1 MeV and φ = −30°, the z→z map element is 1.009 instead of 1.001. The per-particle and three-point kernels share the error; the per-particle one grows the longitudinal phase-space area by 0.7903% per gap.

Run `python TRK-01.py`.

??? example "Reproducer: TRK-01.py"

    [Download TRK-01.py](TRK-01.py)

    ```python
    """TRK-01: The TTF RF gap kernels, default `RfGapTTF` included, apply the particle phase correction with the wrong sign

    The default thin RF gap `RfGapTTF` delays the reference by Δφ₀ = qE₀Lκ(T′ sin φ + S′ cos φ)/(mβ²γ³), with E₀L the gap's
    field integral, φ its RF phase, κ = 2πf/(βc) and T′, S′ the κ-derivatives of the transit-time factors, but adds each
    particle's own correction Δφ to κz instead of subtracting it. A particle delayed more than the reference falls back, so
    κz must decrease by Δφ − Δφ₀, as direct integration through the field confirms. For a 2.5 MeV proton at 402.5 MHz, E₀L =
    0.1 MeV and φ = −30°, the z→z map element is 1.009 instead of 1.001. The per-particle and three-point kernels share the
    error; the per-particle one grows the longitudinal phase-space area by 0.7903% per gap.

    The script builds a symmetric cosine-shaped gap field, fits its transit-time factors T(κ) and T′(κ) as polynomials in κ, and
    tracks particles through half drift, `RfGapTTF` or `RfGapTTF_slow`, and half drift; a uniform field crossed in one
    `RfGapThreePointTTF` step is the third case. It compares the z→z map element (central differences) and the map determinant
    with a DOP853 integration of dK/ds = qE(s)cos(ωt + θ), dt/ds = 1/v through the same field, θ chosen so that the reference
    crosses the gap centre at RF phase φ. The control is the code's own result with only the sign of the particle phase term
    reversed, whose residual falls with (E₀L)² while the code's error falls with E₀L.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.linac import RfGapThreePointTTF, RfGapTTF, RfGapTTF_slow
    from orbit.core.orbit_utils import Polynomial
    from orbit.teapot_base import TPB
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    C = 2.99792458e8
    M = 0.938272013  # GeV, PyORBIT's default (proton) bunch mass
    F = 402.5e6  # Hz
    K0 = 2.5e-3  # GeV kinetic energy
    G = 0.02  # m, field extent [-G/2, G/2]
    PHI = math.radians(-30.0)  # RF phase of the reference at the gap centre
    S_GRID = np.linspace(-G / 2, G / 2, 20001)


    def cos_shape(s):
        """Unit-integral symmetric field shape over [-G/2, G/2]."""
        return np.where(np.abs(s) <= G / 2, math.pi / (2 * G) * np.cos(math.pi * s / G), 0.0)


    def uniform_shape(s):
        return np.where(np.abs(s) <= G / 2, 1.0 / G, 0.0)


    def beta_of(k):
        gam = 1.0 + k / M
        return math.sqrt(1.0 - 1.0 / gam**2)


    def kappa_of(k):
        return 2 * math.pi * F / (C * beta_of(k))


    def poly(coefs):
        p = Polynomial(len(coefs) - 1)
        for i, c in enumerate(coefs):
            p.coefficient(i, float(c))
        return p


    # T(k) = int w cos(k s) ds and T'(k) = dT/dk = -int w s sin(k s) ds for the unit-integral shape w; S = S' = 0 (symmetric)
    W = cos_shape(S_GRID)
    KS = np.linspace(0.8 * kappa_of(K0), 1.2 * kappa_of(K0), 81)
    T_TAB = [np.trapezoid(W * np.cos(k * S_GRID), S_GRID) for k in KS]
    TP_TAB = [-np.trapezoid(W * S_GRID * np.sin(k * S_GRID), S_GRID) for k in KS]
    P_T = poly(np.polynomial.Polynomial.fit(KS, T_TAB, 10).convert().coef)
    P_TP = poly(np.polynomial.Polynomial.fit(KS, TP_TAB, 10).convert().coef)
    P_ZERO = poly([0.0])


    def ttf_model(v, z_in, de_in, kernel):
        """Half drift, thin TTF kernel, half drift; returns (z, dE) at the exit."""
        b = Bunch()
        b.getSyncParticle().kinEnergy(K0)
        b.addParticle(0.0, 0.0, 0.0, 0.0, z_in, de_in)
        TPB.drift(b, G / 2)
        kernel.trackBunch(b, F, v, PHI, P_T, P_ZERO, P_TP, P_ZERO)
        TPB.drift(b, G / 2)
        return b.z(0), b.dE(0)


    def three_point_model(v, z_in, de_in):
        """Uniform field crossed in one three-point step: Em = E0 = Ep = v/G (V/m), half-length dz = G/2."""
        b = Bunch()
        b.getSyncParticle().kinEnergy(K0)
        b.addParticle(0.0, 0.0, 0.0, 0.0, z_in, de_in)
        e = 1e9 * v / G
        TPB.drift(b, G / 2)
        RfGapThreePointTTF().trackBunch(b, G / 2, e, e, e, F, PHI)
        TPB.drift(b, G / 2)
        return b.z(0), b.dE(0)


    def exact_map(shape, v):
        """Direct integration in s with state (tau = c t, K) through E(s) = v*shape(s) GV/m, unit charge."""
        k0 = 2 * math.pi * F / C

        def run(tau0, kin, theta, s_end):
            def rhs(s, y):
                return [1.0 / beta_of(y[1]), v * float(shape(s)) * math.cos(k0 * y[0] + theta)]

            sol = solve_ivp(rhs, (-G / 2, s_end), [tau0, kin], method="DOP853", rtol=1e-13, atol=1e-16)
            return sol.y[0, -1], sol.y[1, -1]

        # theta puts the reference at RF phase PHI when it crosses s = 0, the thin-gap definition of the gap phase
        theta = brentq(
            lambda th: k0 * run(0.0, K0, th, 0.0)[0] + th - PHI, PHI - 3, PHI + 3, xtol=1e-15
        )
        ts, ks = run(0.0, K0, theta, G / 2)

        def fn(z_in, de_in):
            tp, kp = run(
                -z_in / beta_of(K0), K0 + de_in, theta, G / 2
            )  # z = -beta*c*(t_p - t_s): ahead arrives earlier
            return -beta_of(ks) * (tp - ts), kp - ks

        return fn, ks


    def zz(fun, h=1e-6):
        """z->z map element by central differences."""
        return (fun(h, 0.0)[0] - fun(-h, 0.0)[0]) / (2 * h)


    def det2(fun, h=1e-6, he=1e-9):
        """Determinant of the (z, dE) Jacobian by central differences."""
        za, ea = fun(h, 0.0)
        zb, eb = fun(-h, 0.0)
        zc, ec = fun(0.0, he)
        zd, ed = fun(0.0, -he)
        return ((za - zb) / (2 * h)) * ((ec - ed) / (2 * he)) - ((zc - zd) / (2 * he)) * (
            (ea - eb) / (2 * h)
        )


    def reversed_shift(v, z_in, ttf_t, ttf_tp):
        """Change of z_out when the kernel's particle phase term is subtracted instead of added (r = 0, S = S' = 0).

        The kernel computes phase_out = kappa_in*z + pc*T'(kappa_gap)*sin(phi - kappa_in*z) - dphi0, with
        pc = q*E0L*kappa_gap/(m*beta_gap^2*gamma_gap^3) and dphi0 = pc*T'*sin(phi), then z_out = phase_out/kappa_out.
        Reversing the sign of the relative term pc*T'*(sin(psi) - sin(phi)) moves z_out by -2*term/kappa_out; the exit
        drift is linear in z, so the shift carries through it unchanged.
        """
        k_mid = K0 + v * ttf_t(kappa_of(K0)) * math.cos(PHI) / 2
        bg, gg = beta_of(k_mid), 1 + k_mid / M
        kap_g = kappa_of(k_mid)
        k_out = K0 + v * ttf_t(kap_g) * math.cos(PHI)
        pc = v * kap_g / (M * bg**2 * gg**3)
        rel = pc * ttf_tp(kap_g) * (math.sin(PHI - kappa_of(K0) * z_in) - math.sin(PHI))
        return -2 * rel / kappa_of(k_out)


    def uniform_t(k):
        x = k * G / 2
        return math.sin(x) / x


    def uniform_tp(k):
        # dT/dk of sin(k dz)/(k dz), the three-point kernel's T' for a = b = 0
        x = k * G / 2
        return (x * math.cos(x) - math.sin(x)) / (k * k * G / 2)


    reproduced = True
    cases = [
        (
            "RfGapTTF",
            cos_shape,
            lambda v: lambda z, e: ttf_model(v, z, e, RfGapTTF()),
            P_T.value,
            P_TP.value,
        ),
        (
            "RfGapThreePointTTF",
            uniform_shape,
            lambda v: lambda z, e: three_point_model(v, z, e),
            uniform_t,
            uniform_tp,
        ),
    ]
    for name, shape, model, ttf_t, ttf_tp in cases:
        errs = []
        for v in (1e-4, 2e-5):
            fun = model(v)
            code = zz(fun)
            rev = zz(
                lambda z, e, fun=fun, v=v, t=ttf_t, tp=ttf_tp: (
                    fun(z, e)[0] + reversed_shift(v, z, t, tp),
                    0.0,
                )
            )
            exact = zz(exact_map(shape, v)[0])
            errs.append((code - exact, rev - exact))
            print(
                f"{name:18s} E0L={v * 1e3:.2f} MeV z->z: code {code:.6f}, expected {exact:.6f} (direct integration); "
                f"code - exact {code - exact:.3e}, reversed sign - exact {rev - exact:.3e}"
            )
        # The code's error scales with E0L (ratio 5 for the 5x amplitude step), the reversed sign's with E0L^2 (ratio 25)
        reproduced &= all(abs(c) > 10 * abs(r) for c, r in errs)
        reproduced &= 4.0 < errs[0][0] / errs[1][0] < 6.0

    # Phase-space area: (t, E) area is conserved and z = -beta*c*dt, so the exact det(z, dE) is beta_out/beta_in.
    v = 1e-4
    exact_fn, k_ref = exact_map(cos_shape, v)
    det_exact = det2(exact_fn)
    det_slow = det2(lambda z, e: ttf_model(v, z, e, RfGapTTF_slow()))
    print(
        f"RfGapTTF_slow      E0L=0.10 MeV det(z,dE): code {det_slow:.6f}, expected {det_exact:.6f} (direct integration; "
        f"beta_out/beta_in = {beta_of(k_ref) / beta_of(K0):.6f}); area change {100 * (det_slow / det_exact - 1):.4f}% per gap"
    )
    reproduced &= abs(det_slow / det_exact - 1) > 1e-3
    print("DEFECT REPRODUCED" if reproduced else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

