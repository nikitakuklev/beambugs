---
title: "DIA-13: The tune-analysis covariance normalization treats a coupled beam as uncoupled when its covariances are below 10⁻⁸"
hide:
  - toc
---

# DIA-13 — The tune-analysis covariance normalization treats a coupled beam as uncoupled when its covariances are below 10⁻⁸

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [matrix.py#L91-L98](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/matrix.py#L91-L98)<br>[matrix.py#L71-L73](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/matrix.py#L71-L73)<br>[TeapotDiagnosticsNode.py#L213-L259](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/TeapotDiagnosticsNode.py#L213-L259) |

`TeapotTuneAnalysisNode.setNormMatrixFromCovMatrix` and `setNormMatrixFromBunch` choose between plane-by-plane and coupled normalization with `numpy.isclose(off_block, 0.0)`, an absolute tolerance of 10⁻⁸ applied to covariances in m², m·rad and rad². The correct normalization depends on the shape of the beam, not its size, so a beam whose cross-plane covariances are all below 10⁻⁸ is normalized plane by plane however strong its coupling. A matched beam with an x–y coupling rotation of 0.4 rad and eigen-emittances 1 and 2.5 nm·rad gets a normalized covariance whose largest off-diagonal term is 50.72% of the largest diagonal one, and single-turn particle tunes spread over 0.1718–0.8474 and 0.2887–0.5031 instead of the eigentunes 0.2100 and 0.3700. The same beam scaled to 1 and 2.5 µm·rad is normalized exactly.

Run `python DIA-13.py`.

??? example "Reproducer: DIA-13.py"

    [Download DIA-13.py](DIA-13.py)

    ```python
    """DIA-13: The tune-analysis covariance normalization treats a coupled beam as uncoupled when its covariances are below 10⁻⁸

    `TeapotTuneAnalysisNode.setNormMatrixFromCovMatrix` and `setNormMatrixFromBunch` choose between plane-by-plane and coupled normalization with `numpy.isclose(off_block, 0.0)`, an absolute tolerance of 10⁻⁸ applied to covariances in m², m·rad and rad². The correct normalization depends on the shape of the beam, not its size, so a beam whose cross-plane covariances are all below 10⁻⁸ is normalized plane by plane however strong its coupling. A matched beam with an x–y coupling rotation of 0.4 rad and eigen-emittances 1 and 2.5 nm·rad gets a normalized covariance whose largest off-diagonal term is 50.72% of the largest diagonal one, and single-turn particle tunes spread over 0.1718–0.8474 and 0.2887–0.5031 instead of the eigentunes 0.2100 and 0.3700. The same beam scaled to 1 and 2.5 µm·rad is normalized exactly.

    The script builds a coupled, matched covariance Σ = A·diag(ε₁, ε₁, ε₂, ε₂)·Aᵀ, with A a symplectic x–y rotation times Twiss normalizations, and the one-turn map M = A·diag(R(ν₁), R(ν₂))·A⁻¹. It sets the tune node's normalization from Σ, places eight particles on the matched ellipses, applies M once between two tracking calls of the node and reads `tune_1` and `tune_2`, which for a linear map must equal the eigentunes for every particle. It runs eigen-emittances (1, 2.5) µm·rad as the control and (1, 2.5) nm·rad as the case; the case is the control beam scaled by 1/√1000 in every coordinate, so every normalized quantity and every tune must be identical.
    """

    import math
    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.diagnostics import TeapotTuneAnalysisNode
    from orbit.diagnostics.matrix import is_coupled

    NU1, NU2 = 0.21, 0.37  # eigentunes of the one-turn map
    PHI = 0.4  # x-y rotation angle of the coupling, rad


    def twiss_norm(beta, alpha):
        # Columns map normalized (u, u') to (x, x'): x = sqrt(beta) u, x' = (-alpha u + u')/sqrt(beta)
        return np.array([[math.sqrt(beta), 0.0], [-alpha / math.sqrt(beta), 1.0 / math.sqrt(beta)]])


    def coupling_matrix():
        n = np.zeros((4, 4))
        n[:2, :2] = twiss_norm(10.0, 1.2)
        n[2:, 2:] = twiss_norm(6.0, -0.7)
        c, s = math.cos(PHI), math.sin(PHI)
        # Rotation of (x, x') into (y, y') by PHI: symplectic, couples the planes fully
        r = np.array([[c, 0, s, 0], [0, c, 0, s], [-s, 0, c, 0], [0, -s, 0, c]])
        return r @ n


    def rot(nu):
        mu = 2.0 * math.pi * nu
        return np.array([[math.cos(mu), math.sin(mu)], [-math.sin(mu), math.cos(mu)]])


    def run(eps1):
        eps2 = 2.5 * eps1
        a = coupling_matrix()
        cov = a @ np.diag([eps1, eps1, eps2, eps2]) @ a.T
        lam = np.zeros((4, 4))
        lam[:2, :2] = rot(NU1)
        lam[2:, 2:] = rot(NU2)
        m = a @ lam @ np.linalg.inv(a)  # one-turn map; cov = m cov m^T exactly in exact arithmetic

        node = TeapotTuneAnalysisNode("tunes")
        node.setNormMatrixFromCovMatrix(cov)
        v = node.getNormMatrix()[:4, :4]
        ncov = v @ cov @ v.T
        offdiag = np.max(np.abs(ncov - np.diag(np.diag(ncov)))) / np.max(np.abs(np.diag(ncov)))

        bunch = Bunch()
        bunch.mass(0.938272013)
        bunch.getSyncParticle().kinEnergy(1.0)
        for k in range(8):
            ph1, ph2 = 0.7 * k + 0.1, 1.3 * k + 0.5
            u = np.array(
                [
                    math.sqrt(eps1) * math.cos(ph1),
                    -math.sqrt(eps1) * math.sin(ph1),
                    math.sqrt(eps2) * math.cos(ph2),
                    -math.sqrt(eps2) * math.sin(ph2),
                ]
            )
            w = a @ u
            bunch.addParticle(w[0], w[1], w[2], w[3], 0.0, 0.0)
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)  # the analyzer prints two informational notes on stderr
        node.track({"bunch": bunch})
        for i in range(bunch.getSize()):
            w = m @ np.array([bunch.x(i), bunch.xp(i), bunch.y(i), bunch.yp(i)])
            bunch.x(i, w[0])
            bunch.xp(i, w[1])
            bunch.y(i, w[2])
            bunch.yp(i, w[3])
        node.track({"bunch": bunch})
        os.dup2(saved, 2)
        t1 = [bunch.partAttrValue("ParticlePhaseAttributes", i, 2) for i in range(bunch.getSize())]
        t2 = [bunch.partAttrValue("ParticlePhaseAttributes", i, 3) for i in range(bunch.getSize())]
        return is_coupled(cov), np.max(np.abs(cov[:2, 2:])), offdiag, t1, t2


    def main():
        tol = 1e-12
        results = {}
        for label, eps1 in (("control eps=(1, 2.5) um", 1e-6), ("case eps=(1, 2.5) nm", 1e-9)):
            coupled, offblock, offdiag, t1, t2 = run(eps1)
            err = max(max(abs(t - NU1) for t in t1), max(abs(t - NU2) for t in t2))
            results[label] = (coupled, offdiag, err)
            print(
                f"[{label}] max|Sigma off-block| = {offblock:.3e}; is_coupled observed {coupled}, expected True "
                f"(x-y rotation {PHI} rad couples the planes at any scale)"
            )
            print(
                f"[{label}] normalized covariance max off-diagonal / max diagonal: observed {offdiag:.4e}, "
                f"expected 0 (V Sigma V^T = diag(eps1, eps1, eps2, eps2) for the matched beam)"
            )
            print(
                f"[{label}] tune_1 observed min {min(t1):.10f} max {max(t1):.10f}, expected {NU1:.10f}; "
                f"tune_2 observed min {min(t2):.10f} max {max(t2):.10f}, expected {NU2:.10f} "
                f"(eigentunes of the linear one-turn map M = A diag(R(nu1), R(nu2)) A^-1)"
            )
        control_ok = (
            results["control eps=(1, 2.5) um"][0] and results["control eps=(1, 2.5) um"][2] < tol
        )
        case_bad = (not results["case eps=(1, 2.5) nm"][0]) and results["case eps=(1, 2.5) nm"][
            2
        ] > 1e-3
        if not control_ok:
            raise RuntimeError(
                "control failed: the 4-D normalization does not reproduce the eigentunes"
            )
        print("DEFECT REPRODUCED" if case_bad else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found<br><small>strongly coupled beam, normalized covariance and tunes</small> | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small>tunes of a matched coupled beam</small> | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small>rolled small-emittance beam, normalized actions</small> | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

