---
title: "OPT-05: Eigentunes fold fractional tunes above 1/2 to 1 − ν"
hide:
  - toc
---

# OPT-05 — Eigentunes fold fractional tunes above 1/2 to 1 − ν

| | |
|---|---|
| Code | [PyORBIT](../index.md) · Optics |
| Priority | P4 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on PyORBIT commit [`28df660`](https://github.com/PyORBIT-Collaboration/PyORBIT3/commit/28df66018aeb), committed 2026-10-14 |
| Tags | <span class="bb-tag">contract-unsettled</span> [?](../../methodology.md#tags) |
| Source | [`MATRIX_Lattice.py#L150-L159`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/matrix_lattice/MATRIX_Lattice.py#L150-L159)<br>[`matrix.py#L4-L5`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/matrix.py#L4-L5)<br>[`matrix.py#L101-L113`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/py/orbit/diagnostics/matrix.py#L101-L113)<br>[`test_sns_4d.py#L44-L48`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/examples/Diagnostics/Tunes/test_sns_4d.py#L44-L48)<br>[`test_matrix_lattice.py#L38-L56`](https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/28df66018aebe4e2b780b9747f4a2d7b8430a5c0/tests/py/orbit/test_matrix_lattice.py#L38-L56) |

`MATRIX_Lattice.getRingParametersDict` reports the eigentunes `fractional tune 1` and `fractional tune 2` as arccos(Re λ)/2π for two eigenvalues λ of the transverse one-turn matrix, and `TransferMatrixAnalysis.eigtunes` of the diagnostics package uses the same formula. The arccos range limits them to [0, 1/2], so a mode with fractional tune ν > 1/2 is reported as 1 − ν; the eigenvalues alone cannot distinguish the two, and neither method uses the eigenvectors that can. In an uncoupled four-cell FODO ring with plane tunes 0.4051 and 0.8812, both report eigentunes 0.4051 and 0.1188. The code's own uncoupled-ring test requires the eigentunes to equal the plane tunes and passes because its ring has fractional tunes below 1/2.

Run `python OPT-05.py`.

??? example "Reproducer: OPT-05.py"

    [Download OPT-05.py](OPT-05.py)

    ```python
    """OPT-05: Eigentunes fold fractional tunes above 1/2 to 1 − ν

    `MATRIX_Lattice.getRingParametersDict` reports the eigentunes `fractional tune 1` and `fractional
    tune 2` as arccos(Re λ)/2π for two eigenvalues λ of the transverse one-turn matrix, and
    `TransferMatrixAnalysis.eigtunes` of the diagnostics package uses the same formula. The arccos range
    limits them to [0, 1/2], so a mode with fractional tune ν > 1/2 is reported as 1 − ν; the
    eigenvalues alone cannot distinguish the two, and neither method uses the eigenvectors that can. In
    an uncoupled four-cell FODO ring with plane tunes 0.4051 and 0.8812, both report eigentunes 0.4051
    and 0.1188. The code's own uncoupled-ring test requires the eigentunes to equal the plane tunes and
    passes because its ring has fractional tunes below 1/2.

    The script builds three uncoupled four-cell FODO rings of quadrupoles and drifts: one with the
    vertical and one with the horizontal fractional tune above 1/2, and a control with both below 1/2.
    It compares the eigentunes of `getRingParametersDict`, and the `eigtunes` of `TransferMatrixAnalysis`
    for the same one-turn matrix, with the plane tunes that `getRingParametersDict` derives from that
    matrix with the sign of M01 resolving ν against 1 − ν, and with tunes unfolded from the
    eigenvectors' symplectic orientation.
    """

    import os

    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

    import numpy as np
    from orbit.core.bunch import Bunch
    from orbit.diagnostics.matrix import TransferMatrixAnalysis
    from orbit.teapot import (
        DriftTEAPOT,
        QuadTEAPOT,
        TEAPOT_Lattice,
        TEAPOT_MATRIX_Lattice,
    )


    def make_ring(kf, kd):
        lat = TEAPOT_Lattice("ring")
        for c in range(4):
            lat.addNode(QuadTEAPOT(f"qf{c}", 0.25, 2, kf))
            lat.addNode(DriftTEAPOT(f"a{c}", 2.0))
            lat.addNode(QuadTEAPOT(f"qd{c}", 0.5, 2, kd))
            lat.addNode(DriftTEAPOT(f"b{c}", 2.0))
            lat.addNode(QuadTEAPOT(f"qe{c}", 0.25, 2, kf))
        lat.addNode(DriftTEAPOT("d", 1.0))
        lat.initialize()
        return lat


    def unfolded_tunes(M):
        """Mode tunes in [0, 1) from the eigenvectors: of each conjugate pair keep the eigenvalue
        lambda = exp(2 pi i nu) whose eigenvector v has Im(v^H U v) > 0, U the 4x4 symplectic form.
        For a 2x2 block M = [[cos mu + alpha sin mu, beta sin mu], [-gamma sin mu, cos mu - alpha sin mu]]
        with beta > 0, the eigenvalue exp(i mu) has v = (1, (i - alpha)/beta) and v^H U v = 2i/beta,
        so this choice returns mu, the phase whose sine has the sign of M01, as the plane tune does.
        (`orbit.diagnostics.matrix.normalize_eigvec` orients eigenvectors to the conjugate,
        Im(v^H U v) = -2, whose eigenvalue is exp(-i mu).)"""
        w, v = np.linalg.eig(M)
        U = np.kron(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
        nus = [
            (np.angle(w[i]) / (2 * np.pi)) % 1.0
            for i in range(4)
            if np.imag(np.conj(v[:, i]) @ U @ v[:, i]) > 0
        ]
        return sorted(float(x) for x in nus)


    bad = False
    for tag, kf, kd in (
        ("vertical tune above 1/2", 0.95, -1.65),
        ("horizontal tune above 1/2", 0.95, -1.3),
        ("control, both tunes below 1/2", 0.4, -0.5),
    ):
        ref = Bunch()
        ref.mass(0.938272)
        ref.charge(1)
        ref.getSyncParticle().kinEnergy(1.0)  # GeV
        ml = TEAPOT_MATRIX_Lattice(make_ring(kf, kd), ref)
        d = ml.getRingParametersDict()
        nux, nuy, nu1, nu2 = (
            d[k]
            for k in (
                "fractional tune x",
                "fractional tune y",
                "fractional tune 1",
                "fractional tune 2",
            )
        )
        mt = ml.getOneTurnMatrix()
        M = np.array([[mt.get(i, j) for j in range(4)] for i in range(4)])
        unf = unfolded_tunes(M)
        print(
            f"[{tag}] plane tunes {nux:.6f} {nuy:.6f}; eigenvector-unfolded {unf[0]:.6f} {unf[1]:.6f}"
        )
        # Uncoupled ring: mode 1 is the x block and mode 2 the y block, so the eigentunes must equal
        # the plane tunes (the contract of the upstream uncoupled-ring test, to 1e-8)
        print(f"  fractional tune 1: observed {nu1:.6f}, expected {nux:.6f} (plane tune x)")
        print(f"  fractional tune 2: observed {nu2:.6f}, expected {nuy:.6f} (plane tune y)")
        # TransferMatrixAnalysis (orbit.diagnostics.matrix) applied to the same one-turn matrix; its
        # eigtunes come from calc_eigtune, arccos(Re lambda)/2pi, in the same mode order
        t1, t2 = TransferMatrixAnalysis(M).eigtunes
        print(
            f"  TransferMatrixAnalysis.eigtunes: observed {t1:.6f}, {t2:.6f}, expected {nux:.6f}, {nuy:.6f} (plane tunes)"
        )
        if any(abs(a - b) > 1e-8 for a, b in ((nu1, nux), (nu2, nuy), (t1, nux), (t2, nuy))):
            bad = True
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |
| Found<br><small class="bb-scope">source reading and a NumPy probe of the formula</small> | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source reading |
| Confirmed | <span class="bb-nowrap">`claude-sonnet-5-5`</span> | high | <span class="bb-nowrap">2026-11-07</span> | main (`28df66018aeb`), source, serial, CPU, double |

