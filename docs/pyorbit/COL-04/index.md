---
title: "COL-04: TImpedance treats larger z as arriving later, so its multi-turn wake acts ahead of the source or with the wrong sign"
hide:
  - toc
---

# COL-04 — `TImpedance` treats larger z as arriving later, so its multi-turn wake acts ahead of the source or with the wrong sign

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Collective effects |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [TImpedance.cc#L304-L308](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/TImpedance.cc#L304-L308)<br>[TImpedance.cc#L329-L333](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/TImpedance.cc#L329-L333)<br>[TImpedance.cc#L356-L411](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/TImpedance.cc#L356-L411)<br>[TImpedance.cc#L429-L476](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/Impedances/TImpedance.cc#L429-L476)<br>[ImpedanceNode.py#L309-L347](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/impedances/ImpedanceNode.py#L309-L347)<br>[test_fd_TImpedance.py#L94-L97](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/tests/examples/Impedances/test_fd_TImpedance.py#L94-L97) |

`TImpedance` builds its transverse kick from betatron sidebands n ± Q of θ = 2πz/C, with Q the tune and C the ring length, which is consistent only if larger z passes the impedance later. In PyORBIT larger z passes earlier (z = −β₀cΔt), so previous turns enter with the betatron phase of the wrong turn, and no impedance table makes the kick causal. With Chao-convention tables, as the upstream tests label theirs, the kick driven by the source's offset has the wrong sign; with engineering-convention tables the kick acts ahead of the source. For a resonator wake with 150 m memory the kicks deviate from the time-domain wake by 194.1% and 106.1% of its peak; on the z-mirrored bunch they match to 1.609%.

Run `python COL-04.py`.

??? example "Reproducer: COL-04.py"

    [Download COL-04.py](COL-04.py)

    ```python
    """COL-04: `TImpedance` treats larger z as arriving later, so its multi-turn wake acts ahead of the source or with the wrong sign

    `TImpedance` builds its transverse kick from betatron sidebands n ± Q of θ = 2πz/C, with Q the tune and C the ring length, which is consistent only if larger z passes the impedance later. In PyORBIT larger z passes earlier (z = −β₀cΔt), so previous turns enter with the betatron phase of the wrong turn, and no impedance table makes the kick causal. With Chao-convention tables, as the upstream tests label theirs, the kick driven by the source's offset has the wrong sign; with engineering-convention tables the kick acts ahead of the source. For a resonator wake with 150 m memory the kicks deviate from the time-domain wake by 194.1% and 106.1% of its peak; on the z-mirrored bunch they match to 1.609%.

    The script places a cluster of source particles and zero-offset witnesses inside a fixed window of length C. It first checks the turn-ordering relation between two source positions half a ring apart, which holds for every causal wake regardless of convention. It then compares kicks for a resonator wake W(τ) = exp(−τ/τ_d) sin(ω_r τ) with a direct time-domain sum over previous turns, reading the wake's impedance in the Chao and in the engineering convention. The engineering reading on the bunch mirrored in z (z → −z) is the control.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.core.impedances import TImpedance
    from orbit.utils.consts import mass_proton, speed_of_light

    C = 248.0  # ring length and TImpedance phase length (m)
    NB = 128  # longitudinal bins
    Q = 6.23  # betatron tune
    BETA_T = 10.0  # Twiss beta (m); alpha = 0
    EK = 1.0  # proton kinetic energy (GeV)
    MS = 1e10  # macro size
    NS = 50  # source macroparticles at one z
    X0 = 1e-3  # source amplitude (m)
    DZ = C / NB


    def run(parts, Zp, Zm, tune=Q):
        """Track once through a fresh TImpedance; return the xp change of every particle."""
        B = Bunch()
        B.mass(mass_proton)
        B.macroSize(MS)
        B.getSyncParticle().kinEnergy(EK)
        for x, xp, z in parts:
            B.addParticle(x, xp, 0.0, 0.0, z, 0.0)
        T = TImpedance(C, 1, NB, 1, 0)
        T.assignLatFuncs(tune, 0.0, BETA_T, tune, 0.0, BETA_T)
        T.assignImpedance("X", list(Zp), list(Zm))
        T.trackBunch(B)
        return np.array([B.xp(i) for i in range(B.getSize())]), B


    def source(z, w):
        # Betatron phasor w = x - i(beta x' + alpha x), alpha = 0; one turn advances w by exp(+2 pi i Q)
        # (Courant-Snyder one-turn matrix x1 = x cos(mu) + beta x' sin(mu)).
        return [(X0 * w.real, -X0 * w.imag / BETA_T, z)] * NS


    # Two zero-offset anchors fix the bin window to [-C/2, C/2] (TImpedance centres it on the z extrema).
    ANCH = [(0.0, 0.0, -C / 2 * (1 - 1e-9)), (0.0, 0.0, C / 2 * (1 - 1e-9))]

    _, B0 = run(ANCH, [0j] * (NB // 2), [0j] * (NB // 2))
    sp = B0.getSyncParticle()
    V = sp.beta() * speed_of_light
    T0 = C / V
    W0 = 2 * np.pi / T0
    # Kick scale used by TImpedance: macroSize*1e-9*q^2*e*c/(beta*C*(Ek+m)); it multiplies
    # Z (ohm/m) times the dipole sum; f0*kappa_t with kappa_t the time-domain factor below.
    E_SI = 1.602176634e-19
    C2K = MS * 1e-9 * E_SI * speed_of_light / (sp.beta() * C * (sp.kinEnergy() + B0.mass()))

    TD = 150.0 / V  # wake decay time (150 m of flight, multi-turn memory)
    WR = 2 * np.pi * V / 40.0  # wake oscillation, 40 m wavelength
    ZSC = 1e6  # impedance scale (ohm/m per unit Ŵ)


    def what(w):
        # Ŵ(ω) = ∫_0^∞ W(τ) e^{-iωτ} dτ for W(τ) = exp(-τ/TD) sin(WR τ)
        a = 1 / TD + 1j * w
        return WR / (a * a + WR * WR)


    # W > 0 deflects a trailing particle toward the source's offset (Δx' = +κ W x_s), the physical
    # short-range sign. Chao's transverse resonator Z = (c/ω)R_s/(1+iQ_r(ω_r/ω-ω/ω_r)) has Re Z > 0 at
    # resonance and Im Z < 0 below it; with this W that is Z_C(ω) = -i conj(Ŵ(ω)). The engineering
    # convention (e^{jωt}) is its conjugate, Z_E(ω) = i Ŵ(ω).
    READINGS = {
        "Chao Z = -i conj(W^)": lambda w: -1j * np.conj(what(w)) * ZSC,
        "engineering Z = i W^": lambda w: 1j * what(w) * ZSC,
    }


    def zlists(F):
        # Same mapping as FreqDep_TImpedance_Node: Zplus_n = Z(f0 (n+Q)), Zminus_n = Z(f0 (n-Q)).
        return ([F((n + Q) * W0) for n in range(NB // 2)], [F((n - Q) * W0) for n in range(NB // 2)])


    def kref(zs, zw, w, shift):
        """Time-domain multi-turn wake kick on witnesses at zw from the source at zs.

        kick = C2K*T0 * NS*X0 * ZSC * sum_k W(tau_k) Re(w e^{-2 pi i Q k}),
        tau_k = k T0 + (z_s - z_w)/v (z = -beta c dt, so larger z passes earlier), and
        Re(w e^{-2 pi i Q k}) is the source offset k turns earlier. The source is deposited as three
        sub-sources at z_s + m*dz with TImpedance's spline weights (1/8, 3/4, 1/8). `shift` evaluates
        at z_w + shift (TImpedance's half-bin offset, a separate defect, is +dz/2 in its own frame).
        """
        out = np.zeros_like(zw)
        for m, wt in ((-1, 0.125), (0, 0.75), (1, 0.125)):
            for k in range(400):
                tau = k * T0 + ((zs + m * DZ) - (zw + shift)) / V
                Wv = np.where(tau > 0, np.exp(-tau / TD) * np.sin(WR * tau), 0.0)
                out += wt * Wv * np.real(w * np.exp(-2j * np.pi * Q * k))
        return C2K * T0 * NS * X0 * ZSC * out


    bad = False

    # ---- Case 1: turn ordering. A witness at distance D ahead of source A (D > 0) and a witness at
    # D - C from source B = A + C/2 see the same previous passes, shifted by one turn:
    # K_A(w) = K_B(w e^{-2 pi i Q}) for any causal wake. The opposite ordering gives K_B(w e^{+2 pi i Q}).
    zA = -C / 2 + (NB // 4 + 0.5) * DZ
    zB = zA + C / 2
    w = np.exp(0.4j)
    Zp, Zm = zlists(READINGS["Chao Z = -i conj(W^)"])
    for D in (0.3 * C, C / 2 - 3.3, 0.7 * C):
        kA = run(ANCH + source(zA, w) + [(0.0, 0.0, zA + D)], Zp, Zm)[0][-1]
        kBm = run(ANCH + source(zB, w * np.exp(-2j * np.pi * Q)) + [(0.0, 0.0, zB + D - C)], Zp, Zm)[0][
            -1
        ]
        kBp = run(ANCH + source(zB, w * np.exp(+2j * np.pi * Q)) + [(0.0, 0.0, zB + D - C)], Zp, Zm)[0][
            -1
        ]
        err = abs(kA - kBm) / abs(kA)
        print(
            f"turn ordering D = {D:6.1f} m: K_A observed {kA:+.6e}; expected K_B(w e^-2piiQ) = {kBm:+.6e} "
            f"(causal), rel. diff {err:.3e}; K_B(w e^+2piiQ) = {kBp:+.6e}, rel. diff {abs(kA - kBp) / abs(kA):.1e}"
        )
        bad |= err > 1e-6

    # ---- Case 2: resonator wake against the time-domain sum ------------------------------------
    zs = 0.0
    zw = np.linspace(-0.98 * C / 2, 0.98 * C / 2, 401)
    ref = kref(zs, zw, w, shift=DZ / 2)
    pk = np.abs(ref).max()
    for name, F in READINGS.items():
        Zp, Zm = zlists(F)
        k = run(ANCH + source(zs, w) + [(0.0, 0.0, z) for z in zw], Zp, Zm)[0][-len(zw) :]
        err = np.abs(k - ref).max() / pk
        ahead = np.abs(k[zw > zs + 2 * DZ]).max() / pk
        print(
            f"resonator wake, {name}: max|K - K_time-domain|/peak observed {err:.4f}, expected < 0.03 "
            f"(line-sum truncation); max|K| ahead of source / peak {ahead:.3f}, time-domain "
            f"{np.abs(ref[zw > zs + 2 * DZ]).max() / pk:.3f}"
        )
        bad |= err > 0.03

    # Chao reading split by source type: an offset-only source (w = 1) gets the negated kick, an
    # angle-only source (w = -i) the correct one, i.e. K(w) = -K_time-domain(conj w).
    Zp, Zm = zlists(READINGS["Chao Z = -i conj(W^)"])
    for lab, ws in (("offset-only source x'=0", 1.0 + 0j), ("angle-only source x=0", -1j)):
        k = run(ANCH + source(zs, ws) + [(0.0, 0.0, z) for z in zw], Zp, Zm)[0][-len(zw) :]
        r = kref(zs, zw, ws, shift=DZ / 2)
        print(
            f"Chao reading, {lab}: max|K - K_td|/peak {np.abs(k - r).max() / np.abs(r).max():.4f}, "
            f"max|K + K_td|/peak {np.abs(k + r).max() / np.abs(r).max():.4f}; expected max|K - K_td|/peak < 0.03"
        )

    # Control: the same engineering-convention impedance on the bunch mirrored in z (z -> -z) reproduces
    # the time-domain kick, so TImpedance is the correct model for a z axis that points backward.
    Zp, Zm = zlists(READINGS["engineering Z = i W^"])
    k = run(ANCH + source(-zs, w) + [(0.0, 0.0, -z) for z in zw], Zp, Zm)[0][-len(zw) :]
    refm = kref(zs, zw, w, shift=-DZ / 2)  # half-bin offset reverses with the axis
    print(
        f"control, engineering Z on the z-mirrored bunch: max|K - K_time-domain|/peak observed "
        f"{np.abs(k - refm).max() / np.abs(refm).max():.5f}, expected < 0.03"
    )
    ctl_ok = np.abs(k - refm).max() / np.abs(refm).max() < 0.03

    print("DEFECT REPRODUCED" if bad and ctl_ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

