---
title: "OPT-35: Transverse mode dispersion omits the longitudinal-mode divisor"
hide:
  - toc
---

# OPT-35 — Transverse mode dispersion omits the longitudinal-mode divisor

| | |
|---|---|
| Code | [Bmad](../index.md) · Optics |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`calc_emittances_and_twiss_from_sigma_matrix`, mode dispersion, bmad/multiparticle/beam_utils.f90:1573–1585](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L1573-L1585)<br>[Tao beam `eta.a/b`, tao/code/tao_evaluate_a_datum.f90:1362–1379](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L1362-L1379)<br>[beam `etap.a/b`, tao/code/tao_evaluate_a_datum.f90:1414–1431](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/tao/code/tao_evaluate_a_datum.f90#L1414-L1431) |

Extracting dispersion from a six-dimensional beam covariance returns the transverse mode cross-moments without dividing by the longitudinal-mode momentum variance. For an imposed horizontal dispersion of −0.12 m and longitudinal Twiss parameters β_c = 17 m and α_c = 0, the reported value is −0.007059 m instead of −0.12 m. For this dispersion shear, the error scales with γ_c = (1 + α_c²)/β_c, reaches −1.068 m at β_c = 0.4 m and α_c = 1.6, and leaves projected dispersion correct.

Run `python OPT-35.py` with [_bmad.py](../helpers/_bmad.py) and [_dispersion_and_moments.py](../helpers/_dispersion_and_moments.py) in the same directory.

??? example "Reproducer: OPT-35.py"

    [Download OPT-35.py](OPT-35.py)

    ```python
    """OPT-35: Transverse mode dispersion omits the longitudinal-mode divisor

    Extracting dispersion from a six-dimensional beam covariance returns the transverse mode
    cross-moments without dividing by the longitudinal-mode momentum variance. For an imposed
    horizontal dispersion of −0.12 m and longitudinal Twiss parameters β_c = 17 m and α_c = 0,
    the reported value is −0.007059 m instead of −0.12 m. For this dispersion shear, the error
    scales with γ_c = (1 + α_c²)/β_c, reaches −1.068 m at β_c = 0.4 m and α_c = 1.6, and leaves
    projected dispersion correct.

    The script builds positive beam covariances with a known symplectic dispersion shear and executes 84
    parameter cases. It validates projected dispersion and covariance, reconstructs the
    longitudinal-mode divisor from the returned normalizing matrix, and compares the transverse mode
    fields with the imposed dispersion.
    """

    import itertools

    import numpy as np
    from _bmad import compare, verdict
    from _dispersion_and_moments import moment_cases, moments, require, twiss

    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=6)))
    weights = 1 + 0.2 * signs[:, 0] * signs[:, 2] * signs[:, 4]
    j6 = np.kron(np.eye(3), [[0, 1], [-1, 0]])
    longitudinal = list(itertools.product([0.4, 2.5, 17, 60], [0, -0.8, 1.6])) + [(1, 0), (3.56, 1.6)]
    cases = []
    for (beta, alpha), strength, escale in itertools.product(longitudinal, [0, 1, -2], [1, 3]):
        eta = strength * np.array([-0.12, 0.021, 0.075, -0.016])
        # This shear adds dispersion and the compensating z row, preserving the symplectic form.
        shear = np.eye(6)
        shear[:4, 5] = eta
        shear[4, :4] = [-eta[1], eta[0], -eta[3], eta[2]]
        require(shear.T @ j6 @ shear, j6)
        factor = np.zeros((6, 6))
        emits = np.array([4e-7, 2.3e-6, 1.1e-5]) * escale
        for j, args in enumerate(
            [(1.7, -0.6, emits[0]), (6.3, 0.35, emits[1]), (beta, alpha, emits[2])]
        ):
            factor[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = twiss(*args)
        factor = shear @ factor
        cov = factor @ factor.T
        require(np.sort(np.abs(np.linalg.eigvals(cov @ j6)))[::2], emits, atol=0, rtol=3e-13)
        if np.min(np.linalg.eigvalsh(cov)) <= 0:
            raise RuntimeError("Covariance is not positive definite")
        points = signs @ factor.T
        require(moments(points, weights)[1], cov, atol=1e-20, rtol=3e-13)
        cases.append(
            dict(
                species="electron",
                pbase=8e8,
                pnew=8e8,
                points=points,
                beta=beta,
                alpha=alpha,
                eta=eta,
                gamma=(1 + alpha**2) / beta,
                cov=cov,
                emits=emits,
                escale=escale,
            )
        )
    rows = moment_cases(cases, weights)
    defect = False
    max_divided_error = 0.0
    for i, case in enumerate(cases, 1):
        values = rows[f"disp{i}"]
        nmat = np.array([rows[f"nmat{i}_{j}"] for j in range(1, 7)])
        sigma = np.array([rows[f"sigma{i}_{j}"] for j in range(1, 7)])
        require(sigma, case["cov"], atol=1e-20, rtol=3e-13)
        require(values[:4], case["eta"], atol=8e-16, rtol=3e-14)
        mass = rows[f"meta{i}"][0]
        # The two transverse modes must retain their assigned emittances, rather
        # than exchanging identity with the longitudinal mode in the eigensolver.
        require(
            rows[f"emit{i}"][4:6],
            np.asarray(case["emits"])[:2] * case["pbase"] / mass,
            atol=0,
            rtol=2e-7,
        )
        # Wolski's mode cross-moment B_j6 divided by the longitudinal mode B_66.
        numerators = nmat[:4, 4:6] @ nmat[5, 4:6]
        divisor = np.dot(nmat[5, 4:6], nmat[5, 4:6])
        # Check the stored numerator identity before diagnosing its missing divisor.
        require(values[4:8], numerators, atol=0, rtol=1e-14)
        require(values[8], divisor, atol=0, rtol=1e-14)
        divided = numerators / divisor
        max_divided_error = max(max_divided_error, float(np.max(np.abs(divided - case["eta"]))))
        # Eigensystem precision limits reconstructed dispersion to ~2e-8 m in this family.
        require(divided, case["eta"], atol=2e-8, rtol=0)
        if np.all(case["eta"] == 0) or np.isclose(case["gamma"], 1, atol=1e-15, rtol=0):
            require(values[4:8], case["eta"], atol=2e-8, rtol=0)
        defect |= compare(
            f"beta_c={case['beta']:g} m alpha_c={case['alpha']:g} eta_x={np.asarray(case['eta'])[0]:g} scale={case['escale']}, mode dispersion (eta,etap)x,y",
            values[4:8],
            case["eta"],
            atol=2e-8,
            rtol=0,
        )
    print(f"maximum error after B66 division (m and dimensionless): {max_divided_error:.17g}")
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |
| Confirmed<br><small>transverse mode dispersion with analytic longitudinal-mode shears</small> | `gpt-6.1-sol` | xhigh | 2026-10-29 | 20261020-0 (`19b8afb46a40`), source-matched 20261021-0 (d00a8f17e55a) libraries, production, CPU, double, serial |

