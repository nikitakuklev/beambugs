---
title: "TRK-19: Spin precession omits tracked charge-to-mass scaling"
hide:
  - toc
---

# TRK-19 — Spin precession omits tracked charge-to-mass scaling

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Discussed in | [[gh:https://github.com/xsuite/xsuite/issues/860]] |
| Source | [`magnet_spin`, track_magnet_radiation.h:128–139](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_radiation.h#L128-L139)<br>[this rigidity, lines 169–176](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_radiation.h#L169-L176)<br>[Field reconstruction, track_magnet_kick.h:423–427](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_kick.h#L423-L427)<br>[track_magnet.h:104–163](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet.h#L104-L163)<br>[reference-mass momentum, lines 82–92](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_legacy_solenoid_radiation.h#L82-L92)<br>[spin rigidity, lines 145–154](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_legacy_solenoid_radiation.h#L145-L154)<br>[mixed-species ensembles and χ, xtrack.tex:327–350](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L327-L350)<br>[Thomas–BMT equation, lines 6649–6657](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6649-L6657)<br>[UniformSolenoid contract, uniform_solenoid.py:26–36](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/uniform_solenoid.py#L26-L36) |

Tracking a particle whose charge-to-mass ratio differs from the reference gives incorrect spin precession in magnetic elements. The precession generator omits χ = (q/q₀)/(m/m₀), where q and m are particle charge and rest mass and q₀ and m₀ are their reference values. A 1.700 GeV/c proton crossing a 0.3100 T, 0.8000 m solenoid on an equal-mass negative-charge reference exits with vertical spin +0.1218 instead of −0.1218, starting with spin along x. Independent integration of the Thomas–BMT equation and an analytic sector-bend check establish the missing factor. Particles with χ = 1 have the correct normalization.

Run `python TRK-19.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-19.py"

    [Download TRK-19.py](TRK-19.py)

    ```python
    """TRK-19: Spin precession omits tracked charge-to-mass scaling

    Tracking a particle whose charge-to-mass ratio differs from the reference gives incorrect spin precession in magnetic elements. The precession generator omits χ = (q/q₀)/(m/m₀), where q and m are particle charge and rest mass and q₀ and m₀ are their reference values. A 1.700 GeV/c proton crossing a 0.3100 T, 0.8000 m solenoid on an equal-mass negative-charge reference exits with vertical spin +0.1218 instead of −0.1218, starting with spin along x. Independent integration of the Thomas–BMT equation and an analytic sector-bend check establish the missing factor. Particles with χ = 1 have the correct normalization.

    The solenoid test integrates the Thomas–BMT equation in physical SI units with independently specified particle mass, charge and flight time. A sector bend checks transverse-field spin relative to a trajectory exactly on the reference orbit. It uses serial CPU execution with one numerical thread and installed prebuilt kernels, with compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.constants import c, e
    from scipy.integrate import solve_ivp

    MASS = 938272089.43
    G = 1.79284734463
    PC = 1.7e9
    LENGTH = 0.8
    FIELD = 0.31
    np.set_printoptions(precision=17, linewidth=10000)
    RTOL, ATOL = 3e-13, 3e-14
    matches = []


    def record(label, observed, expected, faulty, control):
        derivation = (
            "phi=-G*gamma*h*L"
            if label.startswith("bend")
            else "SI Thomas-BMT integration, Omega_t=-(1+G)*q*B/(m*gamma), T=L/(beta*c)"
        )
        error = float(np.max(np.abs(observed - expected)))
        passed = bool(np.allclose(observed, expected, rtol=RTOL, atol=ATOL))
        predicted = bool(np.allclose(observed, faulty, rtol=RTOL, atol=ATOL))
        if control:
            assert passed, label
        print(
            f"{label}: observed={observed} expected={expected} "
            f"derivation={derivation} max_abs_error={error:.17g} control={control} physical_match={passed} missing_factor_match={predicted}"
        )
        matches.append(passed)


    def axial():
        gamma = np.hypot(PC, MASS) / MASS
        beta = PC / np.hypot(PC, MASS)
        mass_kg = MASS * e / c**2
        # Thomas-BMT for v parallel B: Omega_t = -(q/m)*(1+G)/gamma * B.
        # Integrate in fractional flight time u=t/T, independently of Xtrack's
        # reference momentum and species coordinates.
        flight_time = LENGTH / (beta * c)
        omega_t = -(1 + G) * e * FIELD / (mass_kg * gamma)
        result = solve_ivp(
            lambda u, spin: flight_time * np.cross([0, 0, omega_t], spin),
            (0, 1),
            [1.0, 0.0, 0.0],
            method="DOP853",
            rtol=3e-14,
            atol=3e-15,
        )
        assert result.success
        expected = result.y[:, -1]
        phi = omega_t * flight_time
        np.testing.assert_allclose(expected, [np.cos(phi), np.sin(phi), 0.0], rtol=2e-14, atol=2e-15)
        print("DERIVATION axial: SI Thomas-BMT integration; Omega_t=-(1+G)*q*B/(m*gamma), T=L/(beta*c)")
        print(f"SI_ORACLE phi={phi:.17g} flight_time={flight_time:.17g} spin={expected}")
        for kind in ("uniform", "variable", "spline"):
            # Last case has non-unit charge and mass ratios but chi=1.
            for q0, mr, delta in (
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (2.0, 1.0, 0.0),
                (1.0, 2.0, 0.0),
                (1.0, 0.5, 0.23),
                (0.5, 2.0, 0.0),
            ):
                if kind == "spline" and mr != 1:
                    continue
                p0c = PC / (mr * (1 + delta))
                chi = (1 / q0) / mr
                ks = FIELD * c * q0 / p0c
                particles = xt.Particles(
                    _context=CTX,
                    p0c=p0c,
                    mass0=MASS / mr,
                    q0=q0,
                    mass_ratio=mr,
                    charge_ratio=1 / q0,
                    delta=delta,
                    spin_x=1.0,
                    spin_y=0.0,
                    spin_z=0.0,
                    anomalous_magnetic_moment=G,
                )
                if kind == "uniform":
                    element = xt.UniformSolenoid(
                        _context=CTX,
                        length=LENGTH,
                        ks=ks,
                        edge_entry_active=False,
                        edge_exit_active=False,
                    )
                elif kind == "variable":
                    # On axis, transverse fields of the ramp vanish. All spin
                    # generators commute, so the integral equals the mean field.
                    element = xt.VariableSolenoid(
                        _context=CTX,
                        length=LENGTH,
                        ks_profile=[0.4 * ks, 1.6 * ks],
                        num_multipole_kicks=8,
                    )
                else:
                    # Equal masses isolate this spin factor from SplineBoris's
                    # separate orbital conversion defects. On axis no Lorentz
                    # impulse occurs for either sign of its orbit charge.
                    profile = xt.Spline4(
                        val_start=FIELD, der_start=0.0, val_end=FIELD, der_end=0.0, mean=FIELD
                    )
                    element = xt.SplineBoris(_context=CTX, length=LENGTH, bs=profile, n_steps=16)
                element.track(particles)
                assert particles.state[0] == 1
                np.testing.assert_allclose(
                    [particles.x[0], particles.y[0], particles.px[0], particles.py[0]], 0.0, atol=1e-15
                )
                np.testing.assert_allclose(particles.s[0], LENGTH, rtol=3e-15)
                spin = np.array([particles.spin_x[0], particles.spin_y[0], particles.spin_z[0]])
                np.testing.assert_allclose(np.linalg.norm(spin), 1.0, rtol=0.0, atol=1e-14)
                faulty = np.array([np.cos(phi / chi), np.sin(phi / chi), 0.0])
                record(
                    f"{kind} q0={q0} mr={mr} delta={delta} chi={chi}", spin, expected, faulty, chi == 1
                )


    def bend():
        # At 1+delta=chi the orbit curvature chi*k0/(1+delta) equals h.
        # In the co-rotating frame the exact transverse BMT angle is -G*gamma*h*L.
        # This tests B_perp rather than the solenoid's B_parallel term.
        print("DERIVATION bend: local-frame angle phi=-G*gamma*h*L at 1+delta=chi")
        h, length = 0.11, 0.85
        for qr, mr in ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0)):
            chi = qr / mr
            delta = chi - 1
            gamma = np.hypot(1e9 * (1 + delta), MASS) / MASS
            particles = xt.Particles(
                _context=CTX,
                p0c=1e9,
                mass0=MASS,
                q0=1.0,
                charge_ratio=qr,
                mass_ratio=mr,
                delta=delta,
                spin_x=0.0,
                spin_y=0.0,
                spin_z=1.0,
                anomalous_magnetic_moment=G,
            )
            element = xt.Bend(
                _context=CTX,
                length=length,
                angle=h * length,
                k0=h,
                model="bend-kick-bend",
                edge_entry_active=False,
                edge_exit_active=False,
            )
            element.track(particles)
            np.testing.assert_allclose(
                [particles.x[0], particles.px[0], particles.y[0], particles.py[0]], 0.0, atol=3e-14
            )
            phi = -G * gamma * h * length
            wrong_phi = h * length - (1 + G * gamma) * h * length / chi
            spin = np.array([particles.spin_x[0], particles.spin_y[0], particles.spin_z[0]])
            expected = np.array([np.sin(phi), 0.0, np.cos(phi)])
            faulty = np.array([np.sin(wrong_phi), 0.0, np.cos(wrong_phi)])
            print(f"BEND_ORACLE qr={qr} mr={mr} gamma={gamma:.17g} phi={phi:.17g}")
            record(f"bend qr={qr} mr={mr} chi={chi}", spin, expected, faulty, chi == 1)


    if __name__ == "__main__":
        identity()
        axial()
        bend()
        kernels()
        print(
            f"COMPARISONS total={len(matches)} physical_pass={sum(matches)} physical_fail={len(matches) - sum(matches)}"
        )
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

