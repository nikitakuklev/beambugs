---
title: "TRK-10: Defocusing ChrPlasmaLens has the wrong arrival-time sign term"
hide:
  - toc
---

# TRK-10 — Defocusing `ChrPlasmaLens` has the wrong arrival-time sign term

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-28 on ImpactX 26.10 (`615921c`, committed 2026-10-05); not reproduced on ImpactX development (`f9c2a82`), 2026-11-05 |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1742]], filed from this evidence |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1743]], filed from this evidence |
| Source | [`ChrPlasmaLens.H#L219-L223`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ChrPlasmaLens.H#L219-L223) |

In a defocusing `ChrPlasmaLens` (negative strength `k`), one arrival-time term proportional to (x² + y²)·`ds` keeps the sign of the focusing branch, so the arrival-time coordinate `t` is wrong and the map is not symplectic. Integrating the element's documented Hamiltonian (second order in the transverse variables, exact in `pt`) gives the correct `t`; the observed error equals the sign-flipped term to 2e-15 m. For a proton at γ = 2 with `ds` = 0.4 m, `k` = −2 m⁻², x = 0.01 m and y = −0.015 m, `t` is off by 1.459e-4 m with one slice and 1.713e-4 m with four, and the map's Jacobian M violates the symplectic condition MᵀJM = J by 0.01347. The transverse map and focusing lenses are correct.

Run `python TRK-10.py`.

??? example "Reproducer: TRK-10.py"

    [Download TRK-10.py](TRK-10.py)

    ```python
    """TRK-10: Defocusing `ChrPlasmaLens` has the wrong arrival-time sign term

    In a defocusing `ChrPlasmaLens` (negative strength `k`), one arrival-time term proportional to (x² +
    y²)·`ds` keeps the sign of the focusing branch, so the arrival-time coordinate `t` is wrong and the
    map is not symplectic. Integrating the element's documented Hamiltonian (second order in the
    transverse variables, exact in `pt`) gives the correct `t`; the observed error equals the
    sign-flipped term to 2e-15 m. For a proton at γ = 2 with `ds` = 0.4 m, `k` = −2 m⁻², x = 0.01 m and
    y = −0.015 m, `t` is off by 1.459e-4 m with one slice and 1.713e-4 m with four, and the map's
    Jacobian M violates the symplectic condition MᵀJM = J by 0.01347. The transverse map and focusing
    lenses are correct.

    The script tracks protons through `ChrPlasmaLens` with both signs of `k`, at γ = 2 (`ds` = 0.4 m,
    |`k`| = 2 m⁻²) and γ = 2.7 (`ds` = 0.31 m, |`k`| = 0.7 m⁻²), with one and four slices. It compares
    the exit coordinates with a DOP853 integration of the Hamiltonian, itself checked against the
    closed-form transverse flow, and the `t` error with the sign-flipped term summed over slices; a
    fourth-order finite-difference Jacobian gives the symplecticity residual. The focusing lenses are
    the control and must pass both checks.
    """

    import os

    for variable in (
        "NPROC",
        "PJRT_NPROC",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import numpy as np
    from impactx import ImpactX, elements
    from scipy.integrate import solve_ivp

    SIM = ImpactX()
    SIM.particle_shape = 1
    SIM.space_charge = False
    SIM.diagnostics = False
    SIM.slice_step_diagnostics = False
    SIM.init_grids()
    COLUMNS = [
        "position_x",
        "momentum_x",
        "position_y",
        "momentum_y",
        "position_t",
        "momentum_t",
    ]


    def track(initial, gamma, strength, length, slices=1):
        SIM.beam.clear_particles()
        reference = SIM.beam.ref
        reference.reset()
        reference.set_species("proton")
        reference.set_kin_energy_MeV((gamma - 1) * reference.mass_MeV)
        points = np.atleast_2d(initial).copy()
        SIM.beam.add_n_particles(
            x=points[:, 0].copy(),
            px=points[:, 1].copy(),
            y=points[:, 2].copy(),
            py=points[:, 3].copy(),
            t=points[:, 4].copy(),
            pt=points[:, 5].copy(),
            qm=reference.qm_ratio_SI,
            bunch_charge=1e-15,
        )
        SIM.lattice.clear()
        SIM.lattice.append(elements.ChrPlasmaLens(ds=length, k=strength, nslice=slices))
        SIM.track_particles()
        return SIM.beam.to_df(local=True)[COLUMNS].to_numpy(copy=True)


    def hamiltonian_flow(initial, gamma, strength, length):
        # H = -d - pt/beta + (px^2+py^2)/(2d) + k*(x^2+y^2)/2,
        # d = sqrt(1-2pt/beta+pt^2).  The canonical pair is (t,pt).
        beta = np.sqrt(1 - 1 / gamma**2)

        def equations(_, u):
            x, px, y, py, _t, pt = u
            d = np.sqrt(1 - 2 * pt / beta + pt**2)
            dt = (1 / beta - pt) / d - 1 / beta
            dt += (1 / beta - pt) * (px**2 + py**2) / (2 * d**3)
            return (px / d, -strength * x, py / d, -strength * y, dt, 0)

        solution = solve_ivp(
            equations,
            (0, length),
            initial,
            method="DOP853",
            rtol=3e-14,
            atol=2e-16,
            max_step=length / 32,
            dense_output=True,
        )
        assert solution.success
        return solution


    def symplectic_error(initial, gamma, strength, length, step):
        offsets = np.eye(6) * step
        samples = np.concatenate([initial + factor * offsets for factor in (-2, -1, 1, 2)])
        mapped = track(samples, gamma, strength, length)
        jacobian = (
            (mapped[0:6] - 8 * mapped[6:12] + 8 * mapped[12:18] - mapped[18:24]) / (12 * step)
        ).T
        j = np.kron(np.eye(3), np.array([[0.0, 1.0], [-1.0, 0.0]]))
        return np.max(np.abs(jacobian.T @ j @ jacobian - j))


    bad = False
    try:
        SIM.beam.ref.reset()
        SIM.beam.ref.set_species("proton")
        native_mass = SIM.beam.ref.mass_MeV
        reference_gamma = 1 + 938.27208816 / native_mass
        print(f"native_proton_mass_MeV={native_mass:.15g} reference_gamma={reference_gamma:.17g}")
        cases = [
            (
                "gamma2p7",
                np.array([0.006, 0.013, -0.009, -0.007, 0.02, 0.025]),
                2.7,
                0.7,
                0.31,
            ),
            (
                "gamma2",
                np.array([0.01, 0.02, -0.015, 0.01, 0.03, -0.02]),
                reference_gamma,
                2.0,
                0.4,
            ),
        ]
        for name, initial, gamma, magnitude, length in cases:
            for strength in (magnitude, -magnitude):
                ode = hamiltonian_flow(initial, gamma, strength, length)
                expected = ode.y[:, -1]
                beta = np.sqrt(1 - 1 / gamma**2)
                d = np.sqrt(1 - 2 * initial[5] / beta + initial[5] ** 2)
                omega = np.sqrt(abs(strength) / d)
                trig_c = np.cos(omega * length) if strength > 0 else np.cosh(omega * length)
                trig_s = np.sin(omega * length) if strength > 0 else np.sinh(omega * length)
                # Independent closed form checks that the ODE resolves the transverse flow.
                positions = trig_c * initial[[0, 2]] + trig_s * initial[[1, 3]] / (omega * d)
                momenta = (
                    trig_c * initial[[1, 3]] - np.sign(strength) * omega * d * trig_s * initial[[0, 2]]
                )
                np.testing.assert_allclose(expected[[0, 2]], positions, rtol=2e-13, atol=2e-16)
                np.testing.assert_allclose(expected[[1, 3]], momenta, rtol=2e-13, atol=2e-16)
                for slices in (1, 4):
                    observed = track(initial, gamma, strength, length, slices)[0]
                    np.testing.assert_allclose(
                        observed[[0, 1, 2, 3, 5]],
                        expected[[0, 1, 2, 3, 5]],
                        rtol=2e-13,
                        atol=2e-16,
                    )
                    boundaries = ode.sol(np.arange(slices) * length / slices)
                    radius2_sum = np.sum(boundaries[0] ** 2 + boundaries[2] ** 2)
                    predicted_error = (
                        0.0
                        if strength > 0
                        else (1 / beta - initial[5])
                        * abs(strength)
                        * length
                        * radius2_sum
                        / (2 * d**2 * slices)
                    )
                    error = observed[4] - expected[4]
                    if strength > 0:
                        assert abs(error) < 2e-15, error
                    elif abs(error) >= 2e-15:
                        assert abs(error - predicted_error) < 2e-15, (error, predicted_error)
                        bad = True
                    print(
                        f"{name} k={strength:+g} slices={slices}: t={observed[4]:.15g} ODE={expected[4]:.15g} error={error:.15g} predicted_sign_error={predicted_error:.15g}"
                    )
                for step in (1e-4, 3e-5):
                    residual = symplectic_error(initial, gamma, strength, length, step)
                    print(f"  symplectic_residual step={step:g}: {residual:.12g}")
                    if strength > 0:
                        assert residual < 5e-10
                    elif residual >= 5e-10:
                        assert residual > 1e-4
                        bad = True
    finally:
        SIM.finalize()
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

