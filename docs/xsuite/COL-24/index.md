---
title: "COL-24: 3D FFT Green-function corner subtraction corrupts sparse far-cell fields"
hide:
  - toc
---

# COL-24 — 3D FFT Green-function corner subtraction corrupts sparse far-cell fields

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xfields release [v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7) (`3f06c87`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [FFTSolver3D construction and public Poisson contract, fftsolvers.py:37–114](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/solvers/fftsolvers.py#L37-L114)<br>[primitive_func_3d, lines 309–318](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/solvers/fftsolvers.py#L309-L318)<br>[field-map derivative, interpolated.py:438–470](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/fieldmaps/interpolated.py#L438-L470)<br>[Coulomb convolution and uniform-cell integral, xfields.tex:3830–3896](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L3830-L3896)<br>[TriLinearInterpolatedFieldMap solver construction, interpolated.py:510–518](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/fieldmaps/interpolated.py#L510-L518)<br>[SpaceCharge3D longitudinal scaling, spacecharge.py:110–119](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/spacecharge.py#L110-L119)<br>[upstream PIC regression, test_spacecharge.py:140–206](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/tests/test_spacecharge.py#L140-L206) |

On sparse, strongly anisotropic 3D Poisson meshes, subtraction of integrated Green-function corner values can corrupt weak far-cell electric fields. Eight positive cells carrying 1.000 nC on a 33³ mesh with spacings 100.0 µm, 0.1000 µm and 100.0 µm give a vertical field of 145.5 V/m instead of 184.1 V/m, a 20.97% deficit. Direct volume quadrature of the implemented centered stencil establishes the expected result, while direct native-kernel summation locates the error before the FFT. Smooth Gaussian long-bunch controls remain accurate.

Run `python COL-24.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-24.py"

    [Download COL-24.py](COL-24.py)

    ```python
    """COL-24: 3D FFT Green-function corner subtraction corrupts sparse far-cell fields

    On sparse, strongly anisotropic 3D Poisson meshes, subtraction of integrated Green-function corner values can corrupt weak far-cell electric fields. Eight positive cells carrying 1.000 nC on a 33³ mesh with spacings 100.0 µm, 0.1000 µm and 100.0 µm give a vertical field of 145.5 V/m instead of 184.1 V/m, a 20.97% deficit. Direct volume quadrature of the implemented centered stencil establishes the expected result, while direct native-kernel summation locates the error before the FFT. Smooth Gaussian long-bunch controls remain accurate.

    Compare a positive eight-cell mesh with independent tensor Gauss quadrature of the implemented centered field stencil. Direct native-kernel summation and integration of its exact rounded cell endpoints isolate the cancellation before the FFT. A populated Gaussian long bunch supplies a passing control using an independent Laplace-space integral.
    """

    import itertools
    import json
    from itertools import pairwise

    from _prebuilt import CTX, identity, np, report
    from scipy.constants import epsilon_0
    from scipy.integrate import quad
    from scipy.special import erf
    from xfields.solvers.fftsolvers import FFTSolver3D

    K = 1 / (4 * np.pi * epsilon_0)
    identity()
    cases = {}


    def gauss_field(displacement, spacing, order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        u = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        w = (
            np.prod(
                np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1), axis=-1
            ).ravel()
            / 8
        )
        d = np.asarray(displacement) - u * spacing / 2
        result = []
        for axis, h in enumerate(spacing):
            minus, plus = d.copy(), d.copy()
            minus[:, axis] -= h
            plus[:, axis] += h
            rm, rp = np.linalg.norm(minus, axis=1), np.linalg.norm(plus, axis=1)
            # [1/r(d-h)-1/r(d+h)]/(2h), with the reciprocal difference rationalized.
            result.append(K * np.sum(w * 2 * d[:, axis] / (rm * rp * (rm + rp))))
        return np.array(result)


    def native_field(phi, index, spacing):
        result = []
        for axis, h in enumerate(spacing):
            lo, hi = list(index), list(index)
            lo[axis] -= 1
            hi[axis] += 1
            result.append((phi[tuple(lo)] - phi[tuple(hi)]) / (2 * h))
        return np.array(result)



    for aspect in (100, 300, 1000):
        h = np.array((1e-4, 1e-4 / aspect, 1e-4))
        shape, target = (33, 33, 33), (16, 17, 16)
        sources = list(itertools.product((4, 28), (8, 24), (4, 28)))
        charge = 1e-9
        rho = np.zeros(shape, order="F")
        for source in sources:
            rho[source] = charge / (len(sources) * np.prod(h))
        solver = FFTSolver3D(
            dx=h[0], dy=h[1], dz=h[2], nx=shape[0], ny=shape[1], nz=shape[2], context=CTX
        )
        observed = native_field(solver.solve(rho), target, h)
        expected16 = (
            sum(gauss_field((np.array(target) - s) * h, h, 16) for s in sources) * charge / len(sources)
        )
        expected = (
            sum(gauss_field((np.array(target) - s) * h, h, 24) for s in sources) * charge / len(sources)
        )
        np.testing.assert_allclose(expected16[1], expected[1], rtol=3e-14, atol=0)
        direct = []
        for axis in range(3):
            lo, hi = np.array(target), np.array(target)
            lo[axis] -= 1
            hi[axis] += 1
            phim = sum(
                np.longdouble(solver._gint_rep[tuple(np.abs(lo - s))].real) * np.longdouble(rho[s])
                for s in sources
            )
            phip = sum(
                np.longdouble(solver._gint_rep[tuple(np.abs(hi - s))].real) * np.longdouble(rho[s])
                for s in sources
            )
            direct.append(float((phim - phip) / (2 * np.longdouble(h[axis]))))
        # Integrate the exact binary64 endpoints passed to the primitive as well.
        # Long-double accumulation resolves the small transverse stencil without
        # changing those endpoints, separating grid construction from evaluation.
        nodes, weights = np.polynomial.legendre.leggauss(24)
        nodes = nodes.astype(np.longdouble)
        weights = weights.astype(np.longdouble)
        uu = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
        ww = (
            np.prod(
                np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1), axis=-1
            ).reshape(-1)
            / 8
        )
        grids = [np.arange(35) * v - v / 2 for v in h]

        def endpoint_potential(index, sources=sources, grids=grids, uu=uu, ww=ww, rho=rho):
            total = np.longdouble(0)
            for s in sources:
                lag = np.abs(np.array(index) - s)
                lower = np.array([grids[j][lag[j]] for j in range(3)], dtype=np.longdouble)
                upper = np.array([grids[j][lag[j] + 1] for j in range(3)], dtype=np.longdouble)
                points = (lower + upper) / 2 + uu * (upper - lower) / 2
                integral = np.sum(ww / np.sqrt(np.sum(points * points, axis=1))) * np.prod(
                    upper - lower
                )
                total += np.longdouble(K) * np.longdouble(rho[s]) * integral
            return total

        lower_target, upper_target = list(target), list(target)
        lower_target[1] -= 1
        upper_target[1] += 1
        endpoint_y = float(
            (endpoint_potential(lower_target) - endpoint_potential(upper_target))
            / (2 * np.longdouble(h[1]))
        )
        rec = dict(
            kind="symmetric_positive_sources",
            aspect=aspect,
            spacing=h.tolist(),
            sources=sources,
            target=target,
            observed=observed.tolist(),
            direct_kernel=direct,
            expected=expected.tolist(),
            relative_y_error=float(observed[1] / expected[1] - 1),
            rounded_endpoint_expected_y=endpoint_y,
            endpoint_shift_y=endpoint_y - float(expected[1]),
            oracle_relative_difference=float(expected16[1] / expected[1] - 1),
        )
        assert abs(observed[1] - direct[1]) < 1e-4
        assert abs(endpoint_y - expected[1]) < 1e-5
        assert np.max(np.abs(observed[[0, 2]])) < 1e-4
        print("SPARSE", json.dumps(rec), "derivation=rationalized cell-volume stencil quadrature", flush=True)
        cases[f"sparse aspect={aspect}; cell-volume quadrature Ey V/m"] = (observed[1], expected[1])



    def expected_field(h, probabilities, target, axis, charge, eps):
        hmin = min(h)
        hs = np.array(h) / hmin
        centers = [(np.arange(len(p)) - target[j]) * hs[j] for j, p in enumerate(probabilities)]

        def integral(u):
            v = np.exp(u)
            base, minus, plus = 1.0, 1.0, 1.0
            for j, p in enumerate(probabilities):

                def factor(shift, j=j, p=p):
                    d = centers[j] - shift
                    return (
                        np.dot(p, erf(v * (d + hs[j] / 2)) - erf(v * (d - hs[j] / 2)))
                        * np.sqrt(np.pi)
                        / (2 * v * hs[j])
                    )

                if j == axis:
                    minus = factor(-hs[j])
                    plus = factor(hs[j])
                else:
                    base *= factor(0)
            return (
                K * charge / hmin**2 * 2 / np.sqrt(np.pi) * base * (minus - plus) / (2 * hs[axis]) * v
            )

        # Transform t to log(t*hmin); all cell length scales are resolved explicitly.
        points = sorted(
            set(
                [-28.0, -20.0, -10.0, -5.0, 0.0, 5.0, 12.0, 20.0, 32.0]
                + [-float(np.log(v)) for v in hs]
            )
        )
        # Normalize the physical integral before requesting an absolute tolerance;
        # its units and magnitude vary by many orders between mesh geometries.
        unit_scale = max(abs(integral(u)) for u in np.linspace(-28, 32, 121))
        assert unit_scale > 0
        pieces = [
            quad(lambda u: integral(u) / unit_scale, a, b, epsabs=eps, epsrel=eps, limit=250)
            for a, b in pairwise(points)
        ]
        return unit_scale * sum(v for v, _ in pieces), unit_scale * sum(e for _, e in pieces)


    for h in ((1e-4, 1e-4, 0.1),):
        shape = (33, 33, 33)
        charge = 1e-9
        p = np.exp(-0.5 * ((np.arange(33) - 16) / 5) ** 2)
        p /= p.sum()
        probabilities = [p, p, p]
        rho = charge / np.prod(h) * np.einsum("i,j,k->ijk", p, p, p)
        solver = FFTSolver3D(*h, *shape, context=CTX)
        phi = solver.solve(np.asfortranarray(rho))
        target = (19, 18, 20)
        observed, expected, errors, convergence = [], [], [], []
        for axis in range(3):
            a, b = list(target), list(target)
            a[axis] -= 1
            b[axis] += 1
            observed.append(float((phi[tuple(a)] - phi[tuple(b)]) / (2 * h[axis])))
            oracle, err = expected_field(h, probabilities, target, axis, charge, 2e-11)
            oracle2, err2 = expected_field(h, probabilities, target, axis, charge, 2e-12)
            assert abs(oracle - oracle2) <= 2e-10 * abs(oracle2)
            expected.append(oracle2)
            errors.append(err2)
            convergence.append(abs(oracle - oracle2) / abs(oracle2))
        rec = dict(
            spacing=h,
            target=target,
            observed=observed,
            expected=expected,
            relative_error=(np.array(observed) / expected - 1).tolist(),
            quadrature_error_estimate=errors,
            convergence=convergence,
        )
        np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=0)
        print("GAUSSIAN_CONTROL", json.dumps(rec),
              "derivation=piecewise-constant cells integrated in Laplace space", flush=True)

    report(cases, rtol=1e-7, atol=1e-5)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xfields 0.27.5 (`223b4fa134a1`), installed Python solver, CPU FFT, binary64, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xfields 0.27.5 (`223b4fa134a1`), installed Python solver, CPU FFT, binary64, serial |

