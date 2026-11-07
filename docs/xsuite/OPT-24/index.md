---
title: "OPT-24: 4D polarization inverts the singular invariant-axis direction"
hide:
  - toc
---

# OPT-24 — 4D polarization inverts the singular invariant-axis direction

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on xtrack release [v0.118.2](https://github.com/xsuite/xtrack/tree/v0.118.2) (`ad5e808`) |
| Pull request | [[gh:https://github.com/xsuite/xtrack/pull/923]], fix |
| Discussed in | [[gh:https://github.com/xsuite/xtrack/pull/922]] |
| Source | [_add_spin_polarization, spin.py:135–172](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/spin.py#L135-L172)<br>[twiss_defaults_and_input_preparation.py:173–178](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_defaults_and_input_preparation.py#L173-L178)<br>[twiss_postprocess_and_complem_results.py:257–261](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_postprocess_and_complem_results.py#L257-L261)<br>[Line.twiss4d, line.py:2702–2712](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/line.py#L2702-L2712)<br>[twiss.py:188–192](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L188-L192)<br>[387–405](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L387-L405)<br>[public spin guide:4–6](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/spin_polarization.rst#L4-L6)<br>[manual, xtrack.tex:6731–6755](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L6731-L6755) |

Four-dimensional polarization Twiss can raise a singular-matrix exception in a regular flat electron ring with a vertical invariant spin axis. Its response calculation inverts the full spin matrix at the orbital dispersion eigenvalue one before removing the invariant-axis direction. The physical response perpendicular to that axis is finite: a ring with spin tune 0.2319 has zero transverse dispersion response and a projected matrix condition number of one. The failure also occurs with automatic closed-orbit search. Requesting only `spin=True` succeeds, and finite-RF 6D polarization avoids this energy-mode singularity.

Run `python OPT-24.py` with [_spin_ring.py](../helpers/_spin_ring.py), [_common.py](../helpers/_common.py) and [_prebuilt.py](../helpers/_prebuilt.py) in the same directory.

??? example "Reproducer: OPT-24.py"

    [Download OPT-24.py](OPT-24.py)

    ```python
    """OPT-24: 4D polarization inverts the singular invariant-axis direction

    Four-dimensional polarization Twiss can raise a singular-matrix exception in a regular flat electron ring with a vertical invariant spin axis. Its response calculation inverts the full spin matrix at the orbital dispersion eigenvalue one before removing the invariant-axis direction. The physical response perpendicular to that axis is finite: a ring with spin tune 0.2319 has zero transverse dispersion response and a projected matrix condition number of one. The failure also occurs with automatic closed-orbit search. Requesting only `spin=True` succeeds, and finite-RF 6D polarization avoids this energy-mode singularity.

    The script constructs a flat 32 m electron ring and solves the dispersion spin response
    independently in the two-dimensional plane perpendicular to the vertical axis. It
    compares public 4D polarization calls with automatic and explicit closed spin, while
    spin-only 4D and stable-RF 6D polarization serve as controls. A fixed implementation
    returns finite 4D results and prints NOT REPRODUCED.
    """

    import traceback

    from _spin_ring import G, kernels, matrix, np, report, ring, setup, track_spin


    def main():
        setup()
        line = ring()
        tw = line.twiss4d(spin=True)
        spin = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])
        np.testing.assert_allclose(spin, [0, 1, 0], rtol=0, atol=2e-14)
        np.testing.assert_allclose(track_spin(line, spin), spin, rtol=0, atol=2e-14)
        print(f"spin-only 4D: observed_axis={spin} expected_axis=[0,1,0] derivation=planar symmetry")

        # Match the public spin matrix's perturbation amplitude. The dispersion
        # eigenvalue is 1, but a unit-axis derivative must lie in n's tangent plane.
        A = matrix(line, amplitude=1e-5)
        B = np.eye(3)[:, [0, 2]]
        projected = B.T @ (np.eye(3) - A) @ B
        determinant = np.linalg.det(projected)
        expected_det = 4 * np.sin(np.pi * G * 200) ** 2
        np.testing.assert_allclose(determinant, expected_det, rtol=0, atol=2e-14)
        np.testing.assert_allclose(np.linalg.cond(projected), 1, rtol=0, atol=2e-14)
        p = line.particle_ref.copy()
        p.spin_x, p.spin_y, p.spin_z = [0, 1, 0]
        R = line.get_R_matrix(particle_on_co=p)["R_matrix"]
        R5 = np.delete(np.delete(R, 4, axis=0), 4, axis=1)
        values, vectors = np.linalg.eig(R5)
        index = np.argmin(np.abs(values - 1))
        np.testing.assert_allclose(values[index], 1, rtol=0, atol=2e-14)
        mode = np.insert(vectors[:, index].real, 4, 0)
        mode /= mode[5]
        endpoints = []
        for sign in (-1, 1):
            pp = p.copy()
            # R uses pzeta; d(pzeta)/d(delta)=1 at this reference orbit.
            for key, value in zip(("x", "px", "y", "py", "zeta", "delta"), mode, strict=True):
                setattr(pp, key, sign * 1e-6 * value)
            line.track(pp)
            assert pp.state[0] == 1
            endpoints.append(np.array([pp.spin_x[0], pp.spin_y[0], pp.spin_z[0]]))
        forcing = (endpoints[1] - endpoints[0]) / (2e-6)
        response = np.linalg.solve(projected, B.T @ forcing)
        np.testing.assert_allclose(response, 0, rtol=0, atol=2e-14)
        print(
            f"projected response: observed={response} expected=[0,0] "
            f"determinant={determinant:.17g} expected_determinant={expected_det:.17g} "
            f"condition={np.linalg.cond(projected):.17g} "
            f"full_singular_values={np.linalg.svd(np.eye(3)-A, compute_uv=False)} "
            "derivation=planar symmetry and B.T@(I-A)@B transverse solve"
        )
        cases = {}
        for label, rf, explicit in (
            ("4D automatic", False, False),
            ("4D explicit spin", False, True),
            ("6D stable RF", True, False),
        ):
            test_line = ring(rf=rf)
            options = {}
            if explicit:
                p = test_line.particle_ref.copy()
                p.spin_x, p.spin_y, p.spin_z = [0, 1, 0]
                options["particle_on_co"] = p
            failed = False
            try:
                result = test_line.twiss(
                    method="6d" if rf else "4d", polarization_analysis=True, **options,
                )
            except np.linalg.LinAlgError as exc:
                # Only the historical polarization inverse is a defect verdict;
                # setup errors and failures elsewhere must terminate the run.
                frames = traceback.extract_tb(exc.__traceback__)
                assert str(exc) == "Singular matrix"
                assert any(f.name == "_add_spin_polarization" and f.filename.endswith("/twiss/spin.py") for f in frames)
                if rf:
                    raise
                failed = True
                print(f"{label}: observed=LinAlgError: {exc} expected=finite polarization derivation=regular transverse solve")
            else:
                tune = result.spin_tune_fractional
                np.testing.assert_allclose(tune, G * 200, rtol=0, atol=3e-14)
                assert np.isfinite(result.spin_polarization_eq)
                if not rf:
                    derivative = np.column_stack([
                        result.spin_dn_ddelta_x, result.spin_dn_ddelta_y, result.spin_dn_ddelta_z,
                    ])
                    np.testing.assert_allclose(derivative, 0, rtol=0, atol=2e-14)
                    np.testing.assert_allclose(
                        result.spin_polarization_eq, -8 / (5 * np.sqrt(3)), rtol=0, atol=3e-13,
                    )
                print(
                    f"{label}: observed=finite polarization tune={tune:.17g} "
                    f"expected_tune={G*200:.17g} polarization={result.spin_polarization_eq:.17g} "
                    "derivation=flat-ring BMT tune G*gamma; regular transverse response"
                )
            cases[f"{label} exception; derivation=regular transverse response"] = int(failed), 0
        kernels()
        report(cases, rtol=0, atol=0)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

