---
title: "IO-10: Linear longitudinal matching omits position-momentum correlation"
hide:
  - toc
---

# IO-10 — Linear longitudinal matching omits position-momentum correlation

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xpart main (`2633e52`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`generate_longitudinal_coordinates`, generate_longitudinal.py:371–390](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/longitudinal/generate_longitudinal.py#L371-L390)<br>[line characterization](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/longitudinal/generate_longitudinal.py#L23-L140)<br>[public contract](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/longitudinal/generate_longitudinal.py#L175-L195)<br>[matched-bunch wrapper](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/matched_gaussian.py#L171-L215) |

The linear engine of `generate_longitudinal_coordinates` samples independent Gaussian `zeta` and `delta` with RMS momentum spread σ_δ = σ_z/|β_z|, where σ_z is the RMS bunch length and β_z (`bets0`) the periodic longitudinal beta function, discarding the position-momentum correlation and momentum variance required by a nonzero periodic longitudinal alpha. An independent covariance solve establishes the stationary ellipse. For 3 GeV/c protons on segments with (`bets`, `qs`) = (7 m, 0.045) and (21 m, 0.08), and σ_z = 3×10⁻⁴ m, the sampled bunch-length variance after one turn is 0.8108 of its initial value instead of 1. The matched-Gaussian-bunch wrapper reproduces the mismatch. A weak synchrotron tune of 1.429×10⁻³ still gives a relative variance change of 3.790×10⁻⁵.

Run `python IO-10.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: IO-10.py"

    [Download IO-10.py](IO-10.py)

    ```python
    """IO-10: Linear longitudinal matching omits position-momentum correlation

    The linear engine of `generate_longitudinal_coordinates` samples independent Gaussian `zeta` and
    `delta` with RMS momentum spread σ_δ = σ_z/|β_z|, where σ_z is the RMS bunch length and β_z
    (`bets0`) the periodic longitudinal beta function, discarding the position-momentum correlation and
    momentum variance required by a nonzero periodic longitudinal alpha. An independent covariance solve
    establishes the stationary ellipse. For 3 GeV/c protons on segments with (`bets`, `qs`) = (7 m,
    0.045) and (21 m, 0.08), and σ_z = 3×10⁻⁴ m, the sampled bunch-length variance after one turn is
    0.8108 of its initial value instead of 1. The matched-Gaussian-bunch wrapper reproduces the
    mismatch. A weak synchrotron tune of 1.429×10⁻³ still gives a relative variance change of
    3.790×10⁻⁵.

    The script builds two linear segments with (`bets`, `qs`) = (7 m, 0.045) and (21 m, 0.08) and
    lengths 40 and 50 m for 3 GeV/c protons with σ_z = 3×10⁻⁴ m, and replaces the random normals by 36
    Cartesian Gaussian-quadrature points so the sampled moments are exact; a separate unmodified random
    generation uses 131072 particles and seed 60872. It compares the generated covariance Σ and its
    tracked one-turn transport with the stationary solution of MΣMᵀ = Σ for the one-turn map M in
    canonical coordinates, solved independently of Xtrack's Twiss. Fifteen configurations vary beta sign
    and scale, segment order and count, bunch length, energy and focusing strength and include the
    matched-bunch wrapper; equal-beta segments and ensembles built directly from the stationary
    covariance are controls.
    """

    # Run under a time limit: timeout --kill-after=5s 290s python -B IO-10.py

    from unittest.mock import patch

    import xpart as xp
    from _prebuilt import CTX, identity, kernels, np, xt


    def coordinates(particles):
        order = np.argsort(np.asarray(particles.particle_id))
        return np.array([np.asarray(particles.zeta)[order], np.asarray(particles.pzeta)[order]])


    def oscillator(beta, tune):
        phase = 2 * np.pi * tune
        c, s = np.cos(phase), np.sin(phase)
        return np.array([[c, -beta * s], [s / beta, c]])


    def covariance(points):
        centered = points - np.mean(points, axis=1, keepdims=True)
        return centered @ centered.T / points.shape[1]


    def stationary_covariance(matrix, sigma_z, scale_beta):
        """Solve the nullspace of transport-I on symmetric matrices.

        A diagonal coordinate scaling keeps all entries near one. No Twiss
        parameters or closed-form two-by-two covariance formula enter this oracle.
        """
        units = np.diag([sigma_z, sigma_z / scale_beta])
        inverse_units = np.linalg.inv(units)
        normalized = inverse_units @ matrix @ units
        basis = [
            np.array([[1.0, 0.0], [0.0, 0.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [0.0, 1.0]]),
        ]
        columns = []
        for item in basis:
            residual = normalized @ item @ normalized.T - item
            columns.append(residual[[0, 0, 1], [0, 1, 1]])
        _, singular_values, vh = np.linalg.svd(np.array(columns).T)
        assert singular_values[-1] < 2e-14
        assert singular_values[-2] > 1e-5
        solution = vh[-1] / vh[-1, 0]
        fixed = np.array([[solution[0], solution[1]], [solution[1], solution[2]]])
        assert np.linalg.eigvalsh(fixed).min() > 0
        np.testing.assert_allclose(normalized @ fixed @ normalized.T, fixed, rtol=3e-13, atol=3e-14)
        return units @ fixed @ units.T


    def quadrature():
        # Three-node Gaussian quadrature with integer replication of its weights:
        # E[q]=0, E[q^2]=1, E[q^3]=0, E[q^4]=3. Cartesian products are independent.
        nodes = np.array([-np.sqrt(3), 0.0, 0.0, 0.0, 0.0, np.sqrt(3)])
        q1, q2 = np.meshgrid(nodes, nodes, indexing="ij")
        points = np.array([q1.ravel(), q2.ravel()])
        np.testing.assert_allclose(covariance(points), np.eye(2), rtol=0, atol=4e-16)
        return points


    def make_line(betas, tunes, p0c):
        elements = [
            xt.LineSegmentMap(
                _context=CTX,
                length=40.0 + 10 * i,
                qx=0.22 / len(betas),
                qy=0.34 / len(betas),
                qs=tune,
                bets=beta,
            )
            for i, (beta, tune) in enumerate(zip(betas, tunes))
        ]
        line = xt.Line(elements=elements)
        line.particle_ref = xt.Particles(_context=CTX, p0c=p0c, mass0=xt.PROTON_MASS_EV, q0=1)
        line.build_tracker(_context=CTX)
        assert not line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK
        matrix = np.eye(2)
        for beta, tune in zip(betas, tunes):
            matrix = oscillator(beta, tune) @ matrix
        assert abs(np.linalg.det(matrix) - 1) < 1e-14
        assert abs(np.trace(matrix)) < 2
        return line, matrix


    def particle(zeta, *, line, pzeta=None, delta=None):
        assert (pzeta is None) != (delta is None)
        energy = {"delta": delta} if pzeta is None else {"pzeta": pzeta}
        return xt.Particles(
            _context=CTX,
            p0c=float(line.particle_ref.p0c[0]),
            mass0=line.particle_ref.mass0,
            q0=1,
            zeta=zeta,
            **energy,
        )


    def check_case(label, betas, tunes, sigma_z, p0c=3e9, *, wrapper=False):
        line, matrix = make_line(betas, tunes, p0c)
        twiss = line.twiss(method="6d")
        target = stationary_covariance(matrix, sigma_z, abs(betas[0]))
        # Independent closed form from det(M)=1 and M Sigma M^T=Sigma.
        a, b, c, d = matrix.ravel()
        closed_form = sigma_z**2 * np.array([[1, (d - a) / (2 * b)], [(d - a) / (2 * b), -c / b]])
        # The off-diagonal term tends to zero with the focusing strength; normalize
        # by standard-deviation products to avoid dividing by that small correlation.
        oracle_units = np.sqrt(np.outer(np.diag(target), np.diag(target)))
        np.testing.assert_allclose(
            target / oracle_units, closed_form / oracle_units, rtol=0, atol=5e-14
        )
        expected_tune = np.arccos(np.trace(matrix) / 2) / (2 * np.pi)
        assert abs(twiss.qs - expected_tune) < 2e-11
        np.testing.assert_allclose(twiss.R_matrix[4:, 4:], matrix, rtol=2e-10, atol=2e-11)
        w = twiss.W_matrix[0, 4:, 4:]
        w_covariance = w @ w.T
        w_covariance *= sigma_z**2 / w_covariance[0, 0]
        np.testing.assert_allclose(w_covariance, target, rtol=4e-10, atol=1e-23)

        q = quadrature()
        if wrapper:
            draws = [q[0], q[1], q[0], q[1], q[0], q[1]]
            with patch.object(np.random, "normal", side_effect=draws) as mocked:
                bunch = line.xpart.generate_matched_gaussian_bunch(
                    num_particles=q.shape[1],
                    nemitt_x=1e-9,
                    nemitt_y=2e-9,
                    sigma_z=sigma_z,
                    _context=CTX,
                )
            assert mocked.call_count == 6
            initial = coordinates(bunch)
            delta = np.asarray(bunch.delta).copy()
        else:
            with patch.object(np.random, "normal", side_effect=[q[0], q[1]]) as mocked:
                zeta, delta, matcher = xp.generate_longitudinal_coordinates(
                    line=line, num_particles=q.shape[1], sigma_z=sigma_z, return_matcher=True
                )
            assert matcher is None
            assert mocked.call_count == 2
            np.testing.assert_array_equal(zeta, sigma_z * q[0])
            np.testing.assert_array_equal(delta, sigma_z / abs(twiss.bets0) * q[1])
            bunch = particle(zeta, line=line, delta=delta)
            initial = coordinates(bunch)

        beta0 = float(line.particle_ref.beta0[0])
        # Rationalize the relativistic energy difference to avoid subtracting near equals.
        converted = delta * (2 + delta) / (np.sqrt(1 + beta0**2 * delta * (2 + delta)) + 1)
        conversion_error = float(np.max(np.abs(initial[1] - converted)))
        assert conversion_error < 5e-16 / beta0**2
        before = covariance(initial)
        line.track(bunch)
        final = coordinates(bunch)
        native_error = float(np.max(np.abs(final - matrix @ initial)))
        np.testing.assert_allclose(final, matrix @ initial, rtol=1e-12, atol=2e-18)
        after = covariance(final)
        np.testing.assert_allclose(after, matrix @ before @ matrix.T, rtol=2e-12, atol=2e-24)

        # Construct the independently solved stationary ensemble directly in canonical coordinates.
        stationary_points = np.linalg.cholesky(target) @ q
        control = particle(stationary_points[0], line=line, pzeta=stationary_points[1])
        control_before = covariance(coordinates(control))
        line.track(control)
        control_after = covariance(coordinates(control))
        covariance_units = np.sqrt(np.outer(np.diag(target), np.diag(target)))
        # Scale each covariance entry by its natural units, including zero correlation.
        np.testing.assert_allclose(
            control_before / covariance_units, target / covariance_units, rtol=0, atol=5e-14
        )
        np.testing.assert_allclose(
            control_after / covariance_units, target / covariance_units, rtol=0, atol=5e-14
        )

        # Public normalized construction checks the discovery's W-based alternative.
        normalized = xp.build_particles(
            _context=CTX,
            line=line,
            x_norm=np.zeros(q.shape[1]),
            px_norm=np.zeros(q.shape[1]),
            y_norm=np.zeros(q.shape[1]),
            py_norm=np.zeros(q.shape[1]),
            zeta_norm=q[0],
            pzeta_norm=q[1],
            gemitt_zeta=sigma_z**2 / abs(twiss.bets0),
        )
        normalized_before = covariance(coordinates(normalized))
        line.track(normalized)
        normalized_after = covariance(coordinates(normalized))
        for observed in (normalized_before, normalized_after):
            np.testing.assert_allclose(
                observed / covariance_units, target / covariance_units, rtol=0, atol=5e-10
            )

        law = np.diag([sigma_z**2, (sigma_z / abs(twiss.bets0)) ** 2])
        ratio = float(after[0, 0] / before[0, 0])
        law_ratio = float((matrix @ law @ matrix.T)[0, 0] / law[0, 0])
        correlation = float(target[0, 1] / np.sqrt(target[0, 0] * target[1, 1]))
        alpha_magnitude = abs(target[0, 1]) / np.sqrt(np.linalg.det(target))
        record = dict(
            label=label,
            betas=betas,
            tunes=tunes,
            p0c=p0c,
            sigma_z=sigma_z,
            qs=twiss.qs,
            bets0=twiss.bets0,
            matrix=matrix.tolist(),
            stationary_covariance=target.tolist(),
            generated_covariance=before.tolist(),
            tracked_covariance=after.tolist(),
            population_linear_law=law.tolist(),
            stationary_correlation=correlation,
            abs_alpha=alpha_magnitude,
            variance_ratio=ratio,
            population_linear_ratio=law_ratio,
            control_ratio=float(control_after[0, 0] / control_before[0, 0]),
            control_scaled_covariance_error=float(
                np.max(np.abs(control_after - target) / covariance_units)
            ),
            native_map_error=native_error,
            conversion_error=conversion_error,
            wrapper=wrapper,
            normalized_control_scaled_error=float(
                np.max(np.abs(normalized_after - target) / covariance_units)
            ),
        )
        print(
            "MATCH",
            label,
            "sigma_z",
            sigma_z,
            "qs",
            twiss.qs,
            "abs_alpha",
            alpha_magnitude,
            "rho_required",
            correlation,
            "law_var_z_ratio",
            law_ratio,
            "tracked_ratio",
            ratio,
            "control_ratio",
            record["control_ratio"],
        )
        if abs(correlation) > 0.005:
            assert abs(ratio - 1) > 1e-5
            assert abs(before[0, 1]) < 1e-14 * np.sqrt(before[0, 0] * before[1, 1])
        else:
            assert abs(ratio - law_ratio) < 1e-7
        if label == "beta_scale_1.0":
            print("COVARIANCE expected", target.tolist(), "generated", before.tolist())
        return record


    def random_sample_check():
        line, matrix = make_line([7.0, 21.0], [0.045, 0.08], 3e9)
        sigma_z = 3e-4
        target = stationary_covariance(matrix, sigma_z, 7.0)
        np.random.seed(60872)
        zeta, delta = xp.generate_longitudinal_coordinates(
            line=line, num_particles=131072, sigma_z=sigma_z
        )
        bunch = particle(zeta, line=line, delta=delta)
        before = covariance(coordinates(bunch))
        line.track(bunch)
        after = covariance(coordinates(bunch))
        np.testing.assert_allclose(after, matrix @ before @ matrix.T, rtol=2e-12, atol=2e-24)
        actual_ratio = float(after[0, 0] / before[0, 0])
        rho = float(before[0, 1] / np.sqrt(before[0, 0] * before[1, 1]))
        assert abs(rho) < 0.01
        assert abs(actual_ratio - 1) > 0.05
        result = dict(
            seed=60872,
            num_particles=131072,
            target_covariance=target.tolist(),
            sample_covariance=before.tolist(),
            sample_correlation=rho,
            variance_ratio=actual_ratio,
        )
        print("UNMODIFIED_RANDOM_SAMPLE", result)
        return result


    def main():
        identity(random_seed=60872)
        cases = []
        for scale in [0.25, 1.0, 4.0]:
            cases.append(
                check_case(f"beta_scale_{scale}", [7.0 * scale, 21.0 * scale], [0.045, 0.08], 3e-4)
            )
        for sigma in [3e-5, 3e-3]:
            cases.append(check_case(f"amplitude_{sigma}", [7.0, 21.0], [0.045, 0.08], sigma))
        cases.append(check_case("reordered", [21.0, 7.0], [0.08, 0.045], 3e-4))
        cases.append(check_case("negative_betas", [-7.0, -21.0], [0.045, 0.08], 3e-4))
        for factor in [0.01, 0.1]:
            cases.append(
                check_case(f"weak_tune_{factor}", [7.0, 21.0], [0.045 * factor, 0.08 * factor], 3e-4)
            )
        cases.append(check_case("three_segments", [7.0, 14.0, 25.0], [0.021, 0.034, 0.046], 3e-4))
        cases.append(check_case("low_energy", [7.0, 21.0], [0.045, 0.08], 3e-4, p0c=0.8e9))
        for sign in [1.0, -1.0]:
            cases.append(
                check_case(f"equal_beta_{sign}", [7.0 * sign, 7.0 * sign], [0.045, 0.08], 3e-4)
            )
        cases.append(
            check_case("matched_bunch_wrapper", [7.0, 21.0], [0.045, 0.08], 3e-4, wrapper=True)
        )
        cases.append(check_case("discovery_parameters", [10.0, 30.0], [0.07, 0.09], 1e-4, p0c=7e12))
        random_sample_check()
        print(
            "SWEEP",
            len(cases),
            "maximum_native_map_error",
            max(row["native_map_error"] for row in cases),
            "maximum_canonical_scaled_error",
            max(row["control_scaled_covariance_error"] for row in cases),
            "maximum_W_control_scaled_error",
            max(row["normalized_control_scaled_error"] for row in cases),
            "maximum_conversion_error",
            max(row["conversion_error"] for row in cases),
        )
        kernels()
        print(
            "LONGITUDINAL CONFIRMED: tilted fixed-point covariance is omitted; "
            "all canonical stationary controls pass"
        )
        print("DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | xpart 0.23.23 (`2633e528018a`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | xpart 0.23.23 (`2633e528018a`), installed packages, prebuilt kernels, CPU, serial |

