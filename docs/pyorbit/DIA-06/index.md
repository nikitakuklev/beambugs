---
title: "DIA-06: setNormMatrixFromBunch(dim=6) normalizes in dE while the tune analysis applies the matrix to δ"
hide:
  - toc
---

# DIA-06 — `setNormMatrixFromBunch(dim=6)` normalizes in dE while the tune analysis applies the matrix to δ

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`TeapotDiagnosticsNode.py#L243-L259`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/TeapotDiagnosticsNode.py#L243-L259)<br>[`BunchTuneAnalysis.cc#L111-L120`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTuneAnalysis.cc#L111-L120)<br>[`BunchTuneAnalysis.cc#L60-L62`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTuneAnalysis.cc#L60-L62)<br>[`matrix.py#L71-L88`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/matrix.py#L71-L88) |

`TeapotTuneAnalysisNode.setNormMatrixFromBunch(bunch, dim=6)` builds the normalization matrix from the bunch covariance in the stored coordinates (x, x′, y, y′, z, dE), but `BunchTuneAnalysis` multiplies the matrix by (x, x′, y, y′, z, δ) with δ = dE/(β₀²E₀) and E₀ the total energy. When x or x′ correlates with z or dE, as with dispersion, the normal modes mix in dE with the wrong scale, so the actions and phases are wrong unless β₀²E₀ = 1 GeV. For a proton bunch with dispersion η = 2 m, betatron emittance 1.000e-7 m and δ = ±1e-3, the mean action of the dispersive mode is 1.213e-7 m at 1 GeV kinetic energy, 3.724e-6 m at 0.1 GeV and 2.648e-7 m at 10 GeV.

Run `python DIA-06.py`.

??? example "Reproducer: DIA-06.py"

    [Download DIA-06.py](DIA-06.py)

    ```python
    """DIA-06: `setNormMatrixFromBunch(dim=6)` normalizes in dE while the tune analysis applies the matrix to δ

    `TeapotTuneAnalysisNode.setNormMatrixFromBunch(bunch, dim=6)` builds the normalization matrix from
    the bunch covariance in the stored coordinates (x, x′, y, y′, z, dE), but `BunchTuneAnalysis`
    multiplies the matrix by (x, x′, y, y′, z, δ) with δ = dE/(β₀²E₀) and E₀ the total energy. When x or
    x′ correlates with z or dE, as with dispersion, the normal modes mix in dE with the wrong scale, so
    the actions and phases are wrong unless β₀²E₀ = 1 GeV. For a proton bunch with dispersion η = 2 m,
    betatron emittance 1.000e-7 m and δ = ±1e-3, the mean action of the dispersive mode is 1.213e-7 m at
    1 GeV kinetic energy, 3.724e-6 m at 0.1 GeV and 2.648e-7 m at 10 GeV.

    The script builds 64 proton macroparticles by applying a symplectic dispersion map with η = 2 m to
    an uncoupled sign-factorial set, sets the 6D normalization with `setNormMatrixFromBunch(dim=6)`,
    runs one analysis and compares the mean action of each transverse mode with the eigen-emittances,
    computed with NumPy from the covariance in (x, x′, y, y′, z, δ). It repeats this at 0.1, 1 and
    10 GeV kinetic energy. The control passes the matrix built from the δ covariance to the same
    analyzer.
    """

    import itertools

    import numpy as np
    from orbit.core.bunch import Bunch  # ty: ignore[unresolved-import]
    from orbit.diagnostics import TeapotTuneAnalysisNode
    from orbit.diagnostics.matrix import build_norm_matrix_from_cov

    ETA = 2.0  # m
    U6 = np.kron(np.eye(3), np.array([[0.0, 1.0], [-1.0, 0.0]]))


    def run(kin_energy):
        b = Bunch()
        sp = b.getSyncParticle()
        sp.kinEnergy(kin_energy)
        # The analyzer converts dE to delta = dE / (beta^2 E), with E the total energy.
        k = sp.beta() ** 2 * (sp.kinEnergy() + sp.mass())
        rows = []
        for s in itertools.product((1, -1), repeat=6):
            xb, xpb, y, yp, z0, d = (
                s[0] * 1e-3,
                s[1] * 1e-4,
                s[2] * 2e-3,
                s[3] * 1e-4,
                s[4] * 1e-2,
                s[5] * 1e-3,
            )
            # x = x_b + eta delta, z = z0 + eta x'_b is symplectic in (x, x', y, y', z, delta)
            # (M^T U M = U), so it keeps the eigen-emittances of the uncoupled set.
            row = (xb + ETA * d, xpb, y, yp, z0 + ETA * xpb, d)
            rows.append(row)
            b.addParticle(*row[:5], d * k)
        V = np.array(rows)
        S_delta = V.T @ V / len(V)  # every coordinate has zero mean
        # Eigen-emittances: |Im| of the eigenvalues of S U, one per conjugate pair. The uncoupled set has
        # x: 1e-3 * 1e-4 = 1e-7 m, y: 2e-3 * 1e-4 = 2e-7 m, z: 1e-2 * 1e-3 = 1e-5 m.
        eps = np.sort(np.abs(np.linalg.eigvals(S_delta @ U6).imag))[::2]

        node = TeapotTuneAnalysisNode()
        node.setNormMatrixFromBunch(b, dim=6)
        node.track({"bunch": b})
        d = node.getData(b)
        obs = (float(np.mean(d["action_1"])), float(np.mean(d["action_2"])))

        ctl = TeapotTuneAnalysisNode()
        ctl.setNormMatrix(build_norm_matrix_from_cov(S_delta))
        ctl.track({"bunch": b})
        dc = ctl.getData(b)
        return eps, obs, (float(np.mean(dc["action_1"])), float(np.mean(dc["action_2"])))


    ok = True
    for ke in (1.0, 0.1, 10.0):
        eps, obs, ctl = run(ke)
        # For a normalization N with N S N^T = diag(eps), the mean action (u^2 + u'^2)/2 of mode m is eps_m.
        print(
            f"kinetic energy {ke} GeV: setNormMatrixFromBunch(dim=6) mean action_1 {obs[0]:.6g}, "
            f"action_2 {obs[1]:.6g}; expected {eps[0]:.6g}, {eps[1]:.6g} (eigen-emittances)"
        )
        print(
            f"kinetic energy {ke} GeV: matrix from the delta covariance (control) mean action_1 {ctl[0]:.6g}, "
            f"action_2 {ctl[1]:.6g}"
        )
        ok &= all(abs(c - e) < 1e-9 * e for c, e in zip(ctl, eps[:2]))
        ok &= abs(obs[0] - eps[0]) > 0.01 * eps[0]

    print("DEFECT REPRODUCED" if ok else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

