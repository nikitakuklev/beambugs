---
title: "OPT-26: Full radiation covariance discards complex diffusion correlations"
hide:
  - toc
---

# OPT-26 — Full radiation covariance discards complex diffusion correlations

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [_get_equilibrium_emittance_full, radiation.py:244–357](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/radiation.py#L244-L357)<br>[full radiation-analysis caller, twiss_postprocess_and_complem_results.py:168–230](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_postprocess_and_complem_results.py#L168-L230)<br>[manual balance, xtrack.tex:6064–6094](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6064-L6094) |

Full radiation analysis discards imaginary diffusion correlations before reconstructing equilibrium covariance, although those correlations need not vanish for distinct damped modes. In a 160.0 m, 3.000 GeV electron ring near the difference resonance, a skew quadrupole of integrated strength 1.000×10⁻⁵ m⁻¹ gives a vertical rms size of 4.005 mm instead of 2.569 mm, a 55.90% excess. An independent real-coordinate Lyapunov solve with separately checked photon diffusion establishes the result. Keeping the complex correlations restores agreement. Scalar mode emittances are unchanged by this projection; separated working points have much smaller covariance errors.

Run `python OPT-26.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-26.py"

    [Download OPT-26.py](OPT-26.py)

    ```python
    """OPT-26: Full radiation covariance discards complex diffusion correlations

    Full radiation analysis discards imaginary diffusion correlations before reconstructing equilibrium covariance, although those correlations need not vanish for distinct damped modes. In a 160.0 m, 3.000 GeV electron ring near the difference resonance, a skew quadrupole of integrated strength 1.000×10⁻⁵ m⁻¹ gives a vertical rms size of 4.005 mm instead of 2.569 mm, a 55.90% excess. An independent real-coordinate Lyapunov solve with separately checked photon diffusion establishes the result. Keeping the complex correlations restores agreement. Scalar mode emittances are unchanged by this projection; separated working points have much smaller covariance errors.

    Construct a 160 m electron ring, match its bare tunes and add a weak skew quadrupole before public mean-radiation compensation and full Twiss analysis. Check photon diffusion independently in SI units and solve the stationary covariance in real coordinates with binary64 and 65-digit arithmetic. Compare retained-complex reconstruction, scalar mode emittances, separated tunes and refined tracking differences.
    """

    import importlib
    import json
    import sys

    import mpmath as mp
    from _prebuilt import CTX, identity, kernels, np, report, xt
    from scipy.constants import c, e, epsilon_0, hbar
    from scipy.linalg import solve_discrete_lyapunov
    from scipy.optimize import root

    rad = importlib.import_module("xtrack.twiss.radiation")
    mp.mp.dps = 65
    identity(radiation="mean; full radiation analysis")


    def build_ring():
        elements: list[xt.BeamElement] = [xt.Marker()]
        names = ["start"]
        for i in range(16):
            for name, element in (
                ("qf", xt.Multipole(knl=[0, 0.18])),
                ("da", xt.Drift(length=1.5)),
                ("ba", xt.Bend(length=2, angle=np.pi / 16)),
                ("db", xt.Drift(length=1.5)),
                ("qd", xt.Multipole(knl=[0, -0.18])),
                ("dc", xt.Drift(length=1.5)),
                ("bb", xt.Bend(length=2, angle=np.pi / 16)),
                ("dd", xt.Drift(length=1.5)),
            ):
                elements.append(element)
                names.append(f"{name}{i}")
        elements += [
            xt.Multipole(ksl=[0, 0]),
            xt.Cavity(voltage=6e6, frequency=80 * c / 160, phase=np.pi),
            xt.Marker(),
        ]
        names += ["skew", "rf", "end"]
        line = xt.Line(elements=elements, element_names=names)
        line.particle_ref = xt.Particles(_context=CTX, p0c=3e9, mass0=xt.ELECTRON_MASS_EV, q0=-1)
        line.build_tracker(_context=CTX)
        return line


    def set_quads(line, strengths):
        for i in range(16):
            line[f"qf{i}"].knl[1] = strengths[0]
            line[f"qd{i}"].knl[1] = strengths[1]


    def match_bare(line, gap):
        def residual(strengths):
            set_quads(line, strengths)
            tw = line.twiss(method="4d", chrom=False)
            return np.array([tw.qx - (2.2 + gap / 2), tw.qy - (2.2 - gap / 2)])

        solved = root(residual, [0.18, -0.18], tol=2e-10)
        assert solved.success, solved.message
        err = residual(solved.x)
        assert np.max(np.abs(err)) < 2e-9, err
        return solved.x


    def capture_twiss(line, **kwargs):
        captured = {}

        def profile(frame, event, arg):
            if frame.f_code is rad._get_equilibrium_emittance_full.__code__ and event == "return":
                captured.update(frame.f_locals)

        sys.setprofile(profile)
        try:
            tw = line.twiss(
                radiation_method="full",
                radiation_analysis=True,
                compute_R_element_by_element=True,
                chrom=False,
                **kwargs,
            )
        finally:
            sys.setprofile(None)
        assert captured
        return tw, captured


    def hp_lyapunov(matrix, noise):
        # vec in row-major order gives I - M tensor M; no modal eigenvectors enter.
        m, q = mp.matrix(matrix.tolist()), mp.matrix(noise.tolist())
        operator = mp.matrix(36)
        for i in range(6):
            for j in range(6):
                for a in range(6):
                    for b in range(6):
                        operator[6 * i + j, 6 * a + b] = int(i == a and j == b) - m[i, a] * m[j, b]
        answer = mp.lu_solve(operator, mp.matrix([q[i, j] for i in range(6) for j in range(6)]))
        return np.array([float(v) for v in answer]).reshape(6, 6)


    cases = {}
    for gap, skew in ((0.02, 1e-4), (0.00144, 1e-4), (0.00142, 1e-5)):
        line = build_ring()
        strengths = match_bare(line, gap)
        line["skew"].ksl[1] = skew
        line.configure_radiation(model="mean")
        line.compensate_radiation_energy_loss()
        tw, cap = capture_twiss(line)
        M, Q = cap["RR"], cap["DSigma0"]
        assert np.all(np.abs(np.linalg.eigvals(M)) < 1)
        # Integrate the photon spectrum's second moment independently in SI units.
        # h is curvature per reference momentum, hence gamma0*gamma(s)^4 rather
        # than gamma(s)^5 when closed-orbit momentum changes around the ring.
        mass_kg = tw.particle_on_co.mass0 * e / c**2
        classical_radius = e**2 / (4 * np.pi * epsilon_0 * mass_kg * c**2)
        reduced_compton = hbar / (mass_kg * c)
        gamma0, beta0 = float(tw.gamma0), float(tw.beta0)
        gamma = gamma0 * (1 + beta0 * tw.ptau[:-1])
        props = cap["sr_distrib_properties"]
        curvature = np.hypot(props["hx"][:-1], props["hy"][:-1])
        independent_variance = (
            55
            / (24 * np.sqrt(3))
            * classical_radius
            * reduced_compton
            * gamma0
            * gamma**4
            * curvature**3
            * cap["dl"]
        )
        nz = independent_variance > 0
        photon_variance_error = float(
            np.max(np.abs(independent_variance[nz] / cap["d_delta_sq_ave"][nz] - 1))
        )
        assert photon_variance_error < 3e-14
        energy_axis = np.zeros((len(independent_variance), 6))
        energy_axis[:, 5] = 1
        pulled_kicks = np.linalg.solve(cap["RR_ebe_hat"][:-1], energy_axis[:, :, None])[:, :, 0]
        independent_Q = np.einsum("ni,nj,n->ij", pulled_kicks, pulled_kicks, independent_variance)
        independent_Q_error = float(np.linalg.norm(independent_Q - Q) / np.linalg.norm(Q))
        assert independent_Q_error < 3e-14
        native = tw.eq_beam_covariance_matrix.Sigma[0]
        # The y position row is invariant under the slopes transformation, so the
        # principal vertical size comparison is also in physical coordinates.
        T0 = cap["TT"][0]
        assert np.array_equal(T0[2], np.eye(6)[2])
        print("START_BASIS", np.max(np.abs(T0 - np.eye(6))), flush=True)
        # The implementation stores Sigma_at_start directly. Compare in its own
        # x', y' coordinates to isolate the projection from any basis conversion.
        native_hat = native
        scale = np.sqrt(np.maximum(np.diag(native_hat), 1e-30))
        # Share transverse coordinate scales; a perfectly planar ring has zero
        # vertical quantum excitation and provides no vertical scale of its own.
        scale[2], scale[3] = scale[0], scale[1]
        ms = M * scale[None, :] / scale[:, None]
        qs = Q / scale[:, None] / scale[None, :]
        expected_scaled = solve_discrete_lyapunov(ms, qs, method="direct")
        expected = expected_scaled * scale[:, None] * scale[None, :]
        hp = hp_lyapunov(ms, qs)
        hp_error = np.linalg.norm(hp - expected_scaled) / np.linalg.norm(hp)
        assert hp_error < 3e-11, hp_error
        E = cap["BB_inv"] @ Q @ cap["BB_inv"].T
        lam = np.diag(cap["lam_eig_full"])
        retained = (cap["BB"] @ (E / (1 - lam[:, None] * lam[None, :])) @ cap["BB"].T).real
        repaired_error = np.linalg.norm(
            (retained - expected) / scale[:, None] / scale[None, :]
        ) / np.linalg.norm(hp)
        retained_scaled = retained / scale[:, None] / scale[None, :]
        repaired_residual = np.linalg.norm(retained_scaled - ms @ retained_scaled @ ms.T - qs)
        operator = np.eye(36) - np.kron(ms, ms)
        forward_bound = (
            np.linalg.norm(np.linalg.inv(operator), 2) * repaired_residual / np.linalg.norm(hp)
        )
        print(
            "RETAINED",
            repaired_error,
            "residual_forward_bound",
            forward_bound,
            "scaled_condition",
            np.linalg.cond(operator),
            flush=True,
        )
        assert repaired_error <= 1.1 * forward_bound + 3e-11
        observed_scaled = native_hat / scale[:, None] / scale[None, :]
        rel = np.linalg.norm(observed_scaled - hp) / np.linalg.norm(hp)
        # Q was referred backwards from every radiation location to the start.
        # Forward one-turn noise is M Q M^T. Its stationary solution equals M S M^T.
        physical = M @ expected @ M.T
        extra_checks = None
        if gap == 0.00142 and skew == 1e-5:
            slim = line.twiss(
                radiation_method="full",
                radiation_analysis=True,
                compute_lattice_functions=False,
                chrom=False,
            )
            slim_error = float(
                np.linalg.norm(
                    slim.eq_beam_covariance_matrix.Sigma - tw.eq_beam_covariance_matrix.Sigma
                )
                / np.linalg.norm(tw.eq_beam_covariance_matrix.Sigma)
            )
            assert slim_error < 3e-11
            refined = []
            for multiplier in (0.5, 0.25):
                steps = {
                    key: multiplier * tw.steps_R_matrix[key]
                    for key in ("dx", "dpx", "dy", "dpy", "dzeta", "ddelta")
                }
                tw_refined, cap_refined = capture_twiss(line, steps_R_matrix=steps)
                mr, qr = cap_refined["RR"], cap_refined["DSigma0"]
                mrs = mr * scale[None, :] / scale[:, None]
                qrs = qr / scale[:, None] / scale[None, :]
                sr = (
                    solve_discrete_lyapunov(mrs, qrs, method="direct") * scale[:, None] * scale[None, :]
                )
                refined.append(
                    dict(
                        multiplier=multiplier,
                        steps=steps,
                        observed_sigma_y_m=float(
                            np.sqrt(tw_refined.eq_beam_covariance_matrix.Sigma[0, 2, 2])
                        ),
                        expected_sigma_y_m=float(np.sqrt(sr[2, 2])),
                        variance_relative_error=float(
                            tw_refined.eq_beam_covariance_matrix.Sigma[0, 2, 2] / sr[2, 2] - 1
                        ),
                    )
                )
            extra_checks = dict(
                no_lattice_covariance_relative_difference=slim_error, refined_steps=refined
            )
        rec = dict(
            bare_gap=gap,
            skew_m_inv=skew,
            quad_strengths=strengths.tolist(),
            qx=float(tw.qx),
            qy=float(tw.qy),
            damping=tw.damping_constants_turns.tolist(),
            energy_loss_eV=float(tw.energy_loss),
            covariance_relative_error=rel,
            variance_relative_errors=(np.diag(native_hat) / np.diag(expected) - 1).tolist(),
            sigma_x_observed_m=float(np.sqrt(native_hat[0, 0])),
            sigma_x_expected_m=float(np.sqrt(expected[0, 0])),
            sigma_y_observed_m=float(np.sqrt(native_hat[2, 2])),
            sigma_y_expected_m=float(np.sqrt(expected[2, 2])),
            diffusion_imaginary_fraction=float(np.linalg.norm(E.imag) / np.linalg.norm(E)),
            fixed_point_residual=float(
                np.linalg.norm(observed_scaled - ms @ observed_scaled @ ms.T - qs) / np.linalg.norm(qs)
            ),
            retain_complex_relative_error=repaired_error,
            hp_relative_difference=hp_error,
            retain_complex_residual_forward_bound=forward_bound,
            scaled_lyapunov_condition=float(np.linalg.cond(operator)),
            start_basis_difference=float(np.max(np.abs(T0 - np.eye(6)))),
            independent_photon_variance_relative_error=photon_variance_error,
            independent_Q_relative_error=independent_Q_error,
            extra_checks=extra_checks,
            physical_kick_placement_variance_change=(
                np.diag(physical) / np.diag(expected) - 1
            ).tolist(),
            eigenvalues=[[float(v.real), float(v.imag)] for v in np.linalg.eigvals(M)],
        )
        print("RING", json.dumps(rec), flush=True)
        # Conjugate-mode diffusion entries are real for real symmetric Q, so
        # scalar emittances survive the premature real projection.
        # Use NumPy magnitude as the native scalar formula does: one ulp in
        # |lambda| is amplified by 1/(1-|lambda|^2) under weak damping.
        scalar_complex = np.array([E[i, i + 1] / (1 - np.abs(lam[i])**2) for i in (0, 2, 4)])
        scalar_native = np.array([tw.eq_gemitt_x, tw.eq_gemitt_y, tw.eq_gemitt_zeta])
        np.testing.assert_allclose(scalar_complex.imag, 0, rtol=0, atol=1e-20)
        np.testing.assert_allclose(scalar_native, scalar_complex.real, rtol=2e-14, atol=0)
        print("SCALAR_EMITTANCE_CONTROL", scalar_native.tolist(),
              "expected", scalar_complex.real.tolist(), "derivation=conjugate-mode diffusion")
        eigenvalues = np.linalg.eigvals(M)
        separation = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
        np.fill_diagonal(separation, np.inf)
        assert separation.min() > 1e-6
        assert np.all(tw.damping_constants_turns > 0)
        np.testing.assert_allclose(physical[2, 2], expected[2, 2], rtol=1e-12)
        if gap == 0.02:
            assert abs(native_hat[2, 2] / expected[2, 2] - 1) < 1e-4
        print("VERTICAL_VARIANCE", f"gap={gap:g} skew={skew:g}",
              "observed", native_hat[2, 2], "expected", expected[2, 2],
              "derivation=real-coordinate discrete Lyapunov solve with SI photon diffusion")
        cases[f"vertical variance m^2 gap={gap:g} skew={skew:g}; real Lyapunov oracle"] = (
            native_hat[2, 2], expected[2, 2])
    kernels()
    report(cases, rtol=1e-7, atol=1e-18)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed Python covariance consumer, binary64, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt tracking kernels (compilation forbidden), binary64, CPU, serial |

