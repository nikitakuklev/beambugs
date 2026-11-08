---
title: "DIA-05: BunchTuneAnalysis starts with an uninitialized normalization matrix, so a 4×4 normalization leaves the z and dE columns undefined"
hide:
  - toc
---

# DIA-05 — `BunchTuneAnalysis` starts with an uninitialized normalization matrix, so a 4×4 normalization leaves the z and dE columns undefined

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Diagnostics |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Source | [`BunchTuneAnalysis.cc#L11-L21`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTuneAnalysis.cc#L11-L21)<br>[`BunchTuneAnalysis.hh#L34-L39`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTuneAnalysis.hh#L34-L39)<br>[`BunchTuneAnalysis.cc#L115-L120`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTuneAnalysis.cc#L115-L120)<br>[`TeapotDiagnosticsNode.py#L169-L173`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/TeapotDiagnosticsNode.py#L169-L173)<br>[`TeapotDiagnosticsNode.py#L200-L259`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/TeapotDiagnosticsNode.py#L200-L259)<br>[`BunchTuneAnalysis.cc#L35-L70`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/src/orbit/BunchDiagnostics/BunchTuneAnalysis.cc#L35-L70) |

The `BunchTuneAnalysis` constructor declares a local identity matrix and a local `erase` flag instead of initializing its members, so a new analyzer's normalization matrix holds whatever memory it received. `TeapotTuneAnalysisNode.setNormMatrix` with a 4×4 matrix, used by `setNormMatrixFromBunch` with its default `dim=4` and by `setNormMatrixFromTransferMatrix` for a 4×4 map, writes only the transverse block, so the z and dE columns of the transverse rows keep that memory and enter every normalized coordinate. When the analyzer reuses memory that held −3, a particle advancing by exact tunes 0.2 and 0.3 at actions 1e-7 m and 3e-7 m reports tunes 0.001485 and 0.01832 and actions 5.281e-5 m and 5.614e-5 m. `setNormMatrixFromTwiss` and 6×6 matrices set every element and are unaffected.

Run `python DIA-05.py`.

??? example "Reproducer: DIA-05.py"

    [Download DIA-05.py](DIA-05.py)

    ```python
    """DIA-05: `BunchTuneAnalysis` starts with an uninitialized normalization matrix, so a 4×4 normalization leaves the z and dE columns undefined

    The `BunchTuneAnalysis` constructor declares a local identity matrix and a local `erase` flag
    instead of initializing its members, so a new analyzer's normalization matrix holds whatever memory
    it received. `TeapotTuneAnalysisNode.setNormMatrix` with a 4×4 matrix, used by
    `setNormMatrixFromBunch` with its default `dim=4` and by `setNormMatrixFromTransferMatrix` for a 4×4
    map, writes only the transverse block, so the z and dE columns of the transverse rows keep that
    memory and enter every normalized coordinate. When the analyzer reuses memory that held −3, a
    particle advancing by exact tunes 0.2 and 0.3 at actions 1e-7 m and 3e-7 m reports tunes 0.001485
    and 0.01832 and actions 5.281e-5 m and 5.614e-5 m. `setNormMatrixFromTwiss` and 6×6 matrices set
    every element and are unaffected.

    The script reads the matrix of a freshly constructed analyzer, then frees an analyzer whose matrix
    holds −3 and creates a `TeapotTuneAnalysisNode` until the node's analyzer reuses that memory. It
    sets a 4×4 normalization for an uncoupled one-turn map with βx = 10 m, αx = 0.4, βy = 20 m,
    αy = −0.3 and tunes 0.2 and 0.3, and tracks a particle with actions 1e-7 m and 3e-7 m,
    z = 1 mm and dE = 2 MeV through one exact turn of that map. The control sets the same matrix
    padded to 6×6 with an identity longitudinal block.
    """

    import math

    import numpy as np
    from orbit.core.bunch import Bunch, BunchTuneAnalysis  # ty: ignore[unresolved-import]
    from orbit.diagnostics import TeapotTuneAnalysisNode

    fresh = BunchTuneAnalysis()
    M0 = np.array([[fresh.getNormMatrixElement(i, j) for j in range(6)] for i in range(6)])
    del fresh
    print(
        f"fresh analyzer: {int(np.sum(M0 != np.eye(6)))} of 36 entries differ from the identity "
        f"the constructor intends; diagonal {[float(f'{v:.3g}') for v in np.diag(M0)]}"
    )


    def rotation(beta, alpha, nu):
        """One-turn 2x2 map of an uncoupled plane: M = I cos(mu) + J sin(mu)."""
        mu = 2 * math.pi * nu
        c, s = math.cos(mu), math.sin(mu)
        return np.array([[c + alpha * s, beta * s], [-(1 + alpha**2) / beta * s, c - alpha * s]])


    def norm_block(beta, alpha):
        """Courant-Snyder normalization: (u, u') = (x/sqrt(beta), (alpha x + beta x')/sqrt(beta))."""
        return np.array([[1 / math.sqrt(beta), 0.0], [alpha / math.sqrt(beta), math.sqrt(beta)]])


    def cs_point(beta, alpha, J, phi):
        """Particle of action J and phase phi: x = sqrt(2 J beta) cos(phi), x' = -sqrt(2J/beta)(sin + alpha cos)."""
        return (
            math.sqrt(2 * J * beta) * math.cos(phi),
            -math.sqrt(2 * J / beta) * (math.sin(phi) + alpha * math.cos(phi)),
        )


    BX, AX, NUX, JX = 10.0, 0.4, 0.2, 1e-7
    BY, AY, NUY, JY = 20.0, -0.3, 0.3, 3e-7
    M4 = np.zeros((4, 4))
    M4[:2, :2], M4[2:, 2:] = rotation(BX, AX, NUX), rotation(BY, AY, NUY)
    V4 = np.zeros((4, 4))
    V4[:2, :2], V4[2:, 2:] = norm_block(BX, AX), norm_block(BY, AY)
    V6 = np.eye(6)
    V6[:4, :4] = V4
    X0 = np.array(cs_point(BX, AX, JX, 0.3) + cs_point(BY, AY, JY, 1.1))


    def measure(node):
        b = Bunch()
        b.getSyncParticle().kinEnergy(1.0)
        b.addParticle(*X0, 1e-3, 2e-3)
        node.track({"bunch": b})
        x1 = M4 @ X0
        b.x(0, x1[0])
        b.xp(0, x1[1])
        b.y(0, x1[2])
        b.yp(0, x1[3])
        node.track({"bunch": b})
        d = node.getData(b)
        return [float(d[k][0]) for k in ("tune_1", "tune_2", "action_1", "action_2")]


    STALE = -3.0
    node = None
    for attempt in range(1, 51):
        s = BunchTuneAnalysis()
        for i in range(6):
            for j in range(6):
                s.setNormMatrixElement(i, j, STALE)
        del s
        node = TeapotTuneAnalysisNode()
        if node.tune_calc.getNormMatrixElement(0, 4) == STALE:
            break
    reused = node.tune_calc.getNormMatrixElement(0, 4) == STALE
    print(f"node analyzer reused the freed matrix: {reused} (attempt {attempt})")
    node.setNormMatrix(V4)
    M = node.getNormMatrix()
    print(
        f"after 4x4 setNormMatrix: M[0,4] = {M[0, 4]:.6g}, M[0,5] = {M[0, 5]:.6g}; expected 0 (z, dE do not enter u1)"
    )

    # An exact turn of the map advances each Courant-Snyder phase by 2 pi nu at constant action.
    exp = [NUX, NUY, JX, JY]
    obs = measure(node)
    print(
        "4x4 setNormMatrix: tune_1, tune_2, action_1, action_2 observed "
        + ", ".join(f"{v:.6g}" for v in obs)
        + "; expected "
        + ", ".join(f"{v:.6g}" for v in exp)
    )
    ctl = TeapotTuneAnalysisNode()
    ctl.setNormMatrix(V6)
    obs_c = measure(ctl)
    print(
        "6x6 zero-padded matrix (control): "
        + ", ".join(f"{v:.6g}" for v in obs_c)
        + "; expected "
        + ", ".join(f"{v:.6g}" for v in exp)
    )

    ctl_ok = all(abs(o - e) < 1e-9 * e for o, e in zip(obs_c, exp))
    uninit = not np.array_equal(M0, np.eye(6))
    wrong = reused and abs(obs[0] - NUX) > 1e-3
    print("DEFECT REPRODUCED" if ctl_ok and uninit and wrong else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `claude-opus-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |
| Confirmed | `claude-sonnet-5-5` | high | 2026-11-07 | main (`28df66018aeb`), source, serial, CPU, double |

