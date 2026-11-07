---
title: "OPT-17: Radiation integrals omit sextupole feed-down from the orbit gradient"
hide:
  - toc
---

# OPT-17 — Radiation integrals omit sextupole feed-down from the orbit gradient

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Source | [`_compute_radiation_integrals`, radiation.py:398–417](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L398-L417)<br>[`_add_strengths_to_twiss_res`, strengths.py:31–37](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/strengths.py#L31-L37)<br>[radiation.py:450–453](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L450-L453)<br>[xtrack.tex:6100–6107](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6100-L6107)<br>[xtrack.tex:6259–6283](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6259-L6283)<br>[`Sextupole.k2`, sextupole.py:32–35](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/sextupole.py#L32-L35)<br>[`shift_x`, _common.py:151–155](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/_common.py#L151-L155)<br>[`Magnet`, magnet.py:37–42](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/magnet.py#L37-L42) |

With `radiation_integrals=True`, a sextupole traversed away from its magnetic axis contributes zero to the fourth radiation integral I₄ because the calculation uses its nominal quadrupole coefficient instead of the field gradient at the orbit. The gradient must include k₂(x − a), where k₂ is the sextupole strength, x the horizontal orbit and a the magnetic-axis displacement. A 0.2000 m sextupole with k₂ = 100.0 m⁻³, a = 5.000 mm and entrance dispersion 0.2000 m returns zero instead of −5.042 × 10⁻⁵ m⁻¹, as independent orbit and dispersion integration establishes. In a separate 132.0 m periodic ring, the missing sextupole term makes the horizontal damping partition 0.005686% too large.

Run `python OPT-17.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-17.py"

    [Download OPT-17.py](OPT-17.py)

    ```python
    """OPT-17: Radiation integrals omit sextupole feed-down from the orbit gradient

    With `radiation_integrals=True`, a sextupole traversed away from its magnetic axis contributes zero to the fourth radiation integral I₄ because the calculation uses its nominal quadrupole coefficient instead of the field gradient at the orbit. The gradient must include k₂(x − a), where k₂ is the sextupole strength, x the horizontal orbit and a the magnetic-axis displacement. A 0.2000 m sextupole with k₂ = 100.0 m⁻³, a = 5.000 mm and entrance dispersion 0.2000 m returns zero instead of −5.042 × 10⁻⁵ m⁻¹, as independent orbit and dispersion integration establishes. In a separate 132.0 m periodic ring, the missing sextupole term makes the horizontal damping partition 0.005686% too large.

    The script compares two algebraically identical field polynomials and integrates an independent orbit, dispersion and I₄ system with SciPy. A constant-gradient magnet checks the numerical quadrature and gradient normalization. It uses deterministic serial CPU tracking with prebuilt kernels and compilation forbidden.
    """

    # ruff: noqa: I001 -- runtime must cap threads before NumPy is imported.
    from _prebuilt import CTX, identity, kernels, xt
    import numpy as np
    from scipy.integrate import solve_ivp

    LENGTH, OFFSET, K2 = 0.2, 0.005, 100.0


    def oracle(k0, k1, k2, offset, n):
        def rhs(_s, state):
            x, px, dispersion, dp, _integral = state
            u = x - offset
            field = k0 + k1 * u + k2 * u * u / 2
            gradient = k1 + k2 * u
            pz = np.sqrt(1 - px * px)
            # Canonical px=Px/p0, fixed p0 and delta=0. Differentiate x'=px/pz
            # with respect to delta to get D'=(Dp-px)/(1-px^2)^(3/2).
            # For a planar straight element the curvature is exactly field at delta=0.
            return [
                px / pz,
                -field,
                (dp - px) / pz**3,
                -gradient * dispersion,
                2 * gradient * field * dispersion,
            ]

        solution = solve_ivp(
            rhs,
            (0.0, LENGTH),
            [0.0, 0.0, 0.2, 0.0, 0.0],
            method="DOP853",
            rtol=2e-13,
            atol=1e-16,
            dense_output=True,
            max_step=LENGTH / 32,
        )
        assert solution.success
        s = np.arange(n + 1) * LENGTH / n
        x, px, dispersion, _dp, _ = solution.sol(s)
        u = x[:-1] - offset
        gradient = k1 + k2 * u
        # The reported integral estimates curvature from the change of slope across
        # each piece, then multiplies by entrance dispersion. Apply that same
        # quadrature to the independent orbit to separate resolution from gradient.
        slope = px / np.sqrt(1 - px**2)
        mid_slope = (slope[:-1] + slope[1:]) / 2
        curvature = -np.diff(slope) / (LENGTH / n) / (1 + mid_slope**2) ** 1.5
        discrete = np.sum(2 * gradient * curvature * dispersion[:-1]) * LENGTH / n
        return solution.y[:, -1], discrete


    def run(n, k0, k1, k2, offset, *, sextupole=False):
        cls = xt.Sextupole if sextupole else xt.Magnet
        strengths = dict(k2=k2) if sextupole else dict(k0=k0, k1=k1, k2=k2)
        elements = [
            cls(
                length=LENGTH / n,
                **strengths,
                shift_x=offset,
                model="drift-kick-drift-exact",
                integrator="yoshida-6",
                num_multipole_kicks=7,
                edge_entry_active=False,
                edge_exit_active=False,
            )
            for _ in range(n)
        ]
        line = xt.Line(elements=elements)
        line.particle_ref = xt.Particles(p0c=3e9, mass0=xt.ELECTRON_MASS_EV, q0=-1)
        line.build_tracker(_context=CTX)
        # A single open magnet has no equilibrium. Only local integral quantities
        # are used; the unrelated equilibrium sqrt can legitimately be undefined.
        with np.errstate(invalid="ignore"):
            tw = line.twiss(method="4d", betx=3.0, bety=4.0, dx=0.2, dpx=0.0, radiation_integrals=True)
        particle = line.particle_ref.copy()
        line.track(particle)
        return tw, np.array([particle.x[0], particle.px[0]])


    identity()
    bad = []
    for n in (128, 512):
        shifted, orbit_a = run(n, 0.0, 0.0, K2, OFFSET, sextupole=True)
        expanded, orbit_b = run(n, K2 * OFFSET**2 / 2, -K2 * OFFSET, K2, 0.0)
        expected, discrete = oracle(0.0, 0.0, K2, OFFSET, n)
        control, orbit_c = run(n, K2 * OFFSET**2 / 2, -K2 * OFFSET, 0.0, 0.0)
        control_expected, control_discrete = oracle(K2 * OFFSET**2 / 2, -K2 * OFFSET, 0.0, 0.0, n)
        print(
            f"shifted sextupole n={n}: observed_I4={shifted.rad_int_i4:.16g} "
            f"expected_I4={expected[4]:.16g}; "
            "derivation=integral 2 K2(x-a) [K2(x-a)^2/2] D ds"
        )
        print(
            f"field-representation difference n={n}: "
            f"observed_I4_difference={shifted.rad_int_i4 - expanded.rad_int_i4:.16g} "
            "expected=0; derivation=K2(x-a)^2/2=K2*x^2/2-K2*a*x+K2*a^2/2"
        )
        print(
            f"orbit-equivalence control n={n}: observed_max_error={np.max(abs(orbit_a - orbit_b)):.16g} "
            "expected=0"
        )
        print(
            f"independent orbit control n={n}: observed={orbit_a.tolist()} "
            f"expected={expected[:2].tolist()}; derivation=Lorentz ODE"
        )
        print(
            f"constant-gradient control n={n}: observed_I4={control.rad_int_i4:.16g} "
            f"expected_I4_same_quadrature={control_discrete:.16g}; "
            "derivation=sum(2 k1 curvature_j D(s_j) ds) from independent orbit"
        )
        print(
            f"quadrature n={n}: sextupole_expected_discrete={discrete:.16g} "
            f"constant_gradient_expected_continuum={control_expected[4]:.16g}"
        )
        assert np.max(abs(orbit_a - orbit_b)) < 2e-15
        assert np.max(abs(orbit_a - expected[:2])) < 2e-14
        assert abs(shifted.dx[-1] - expected[2]) < 2e-9
        assert abs(control.rad_int_i4 - control_discrete) < 5e-14
        assert np.isfinite(shifted.rad_int_i4) and np.isfinite(expanded.rad_int_i4)
        bad.append(abs(shifted.rad_int_i4 - expected[4]) > 0.99 * abs(expected[4]))
        bad.append(abs(shifted.rad_int_i4 - expanded.rad_int_i4) > 4e-5)
    kernels()
    print("DEFECT REPRODUCED" if all(bad) else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-05 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |

