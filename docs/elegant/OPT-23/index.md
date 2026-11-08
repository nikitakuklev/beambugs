---
title: "OPT-23: moments_output mode emittances normalize the (s, δ) pair with the transverse sign"
hide:
  - toc
---

# OPT-23 — moments_output mode emittances normalize the (s, δ) pair with the transverse sign

| | |
|---|---|
| Code | [Elegant](../index.md) · Optics |
| Priority | P2 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-04 on Elegant commit [`7514db3`](https://github.com/rtsoliday/elegant/commit/7514db36f764), committed 2026-11-04 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`NormalizeEigenvectors`, src/moments.c:1123–1136](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1123-L1136)<br>[`computeNaturalEmittances`, src/moments.c:1054–1055](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L1054-L1055)<br>[`e1`, `e2`, `e3`, emittances of modes 1 to 3, src/moments.c:111–113](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/moments.c#L111-L113)<br>[changing the time sign for symplecticity, src/coupled_twiss.c:381–385](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/src/coupled_twiss.c#L381-L385)<br>[canonical variables are (x, q_x, y, q_y, −s, δ), doc/elegant.tex:987](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L987)<br>[the equilibrium moments, doc/elegant.tex:5613](https://github.com/rtsoliday/elegant/blob/0355dd6aca12c861971fbf8f3ae78a0ded991734/doc/elegant.tex#L5613) |

Elegant's `moments_output` normalizes its normal modes with the same symplectic sign for all coordinate pairs, although the canonical longitudinal pair is (−s, δ), where s is path length and δ is relative momentum deviation. With transverse–longitudinal mixing, this gives wrong mode emittances; without intrabeam scattering, the covariance still solves the exported linear equilibrium equation. In a 2.4 GeV, 70 m ring with a 1.5 MV cavity, the covariance's paraxial eigen-emittances are 5.448 and 51.45 μm, but Elegant reports 5.418 and 51.17 μm (−0.5564% and −0.5555%). Changing the ring's starting point gives a −0.7020% error in the first mode.

Run `python OPT-23.py` with [_elegant.py](../helpers/_elegant.py) in the same directory.

??? example "Reproducer: OPT-23.py"

    [Download OPT-23.py](OPT-23.py)

    ```python
    """OPT-23: moments_output mode emittances normalize the (s, δ) pair with the transverse sign

    Elegant's `moments_output` normalizes its normal modes with the same symplectic sign for all
    coordinate pairs, although the canonical longitudinal pair is (−s, δ), where s is path length
    and δ is relative momentum deviation. With transverse–longitudinal mixing, this gives wrong mode
    emittances; without intrabeam scattering, the covariance still solves the exported linear
    equilibrium equation. In a 2.4 GeV, 70 m ring with a 1.5 MV cavity, the covariance's paraxial
    eigen-emittances are 5.448 and 51.45 μm, but Elegant reports 5.418 and 51.17 μm (−0.5564% and
    −0.5555%). Changing the ring's starting point gives a −0.7020% error in the first mode.

    The script runs serial Elegant `moments_output` (matched, equilibrium, radiation) on a 2.4 GeV,
    70 m ring of 20 FODO cells with radiating CSBENDs and a 1.5 MV RF cavity, from two starting points
    of the same ring, with `matrix_output` writing the one-turn matrix M and diffusion matrix D. It
    solves Σ = MΣMᵀ + D independently with SciPy, checks it against Elegant's equilibrium covariance,
    and takes the expected mode emittances as the moduli of the eigenvalues of ΣS with
    S = diag(J, J, −J), J = [[0, 1], [−1, 0]]: the minus sign makes (−s, δ) the canonical pair. It
    compares these with the reported e1 and e3 (e2 is zero in this flat ring), and recomputes the
    mode emittances from the eigenvectors of M normalized with diag(J, J, J), as the code does, and with
    diag(J, J, −J).
    """

    import sys

    import numpy as np
    from _elegant import run, verdict
    from scipy.linalg import block_diag, solve_discrete_lyapunov

    N_CELLS = 20
    LATTICE = f"""F: QUAD,L=0.3,K1=0.9
    G: QUAD,L=0.3,K1=-0.85
    D: DRIF,L=0.65
    E: DRIF,L=0.4
    B: CSBEND,L=1.2,ANGLE={2 * np.pi / N_CELLS!r},N_SLICES=40,EXPAND_HAMILTONIAN=0,SYNCH_RAD=1,ISR=1
    C: LINE=(F,D,B,E,G,D)
    RF: RFCA,VOLT=1500000,FREQ={80 * 299792458 / 70!r},PHASE=170,CHANGE_P0=0
    A: LINE=(RF,{N_CELLS}*C)
    Z: LINE=(F,D,B,E,G,D,RF,{N_CELLS - 1}*C)
    """
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    RTOL = 1e-5  # the expected values are exact eigen-emittances of a covariance matched to ~1e-13


    def mode_emittances(m, sigma, s):
        """Emittance of each oscillation mode from the left eigenvectors v of the one-turn matrix:
        (v^H sigma v) / |Im(v^H s v)|, the normalization moments_output uses with its own s."""
        values, vectors = np.linalg.eig(m.T)
        found = []
        for k in np.where(values.imag > 0)[0]:
            v = vectors[:, k]
            found.append((v.conj() @ sigma @ v).real / abs((v.conj() @ s @ v).imag))
        return np.sort(found)


    def eigen_emittances(sigma, s):
        """Mode emittances: moduli of the eigenvalues +-i eps_k of sigma @ s, in increasing order."""
        values = np.linalg.eigvals(sigma @ s)
        return np.sort(np.abs(values.imag))[::2]


    defect = False
    for line in ("A", "Z"):
        commands = (
            f"&run_setup lattice=ring.lte, use_beamline={line}, p_central_mev=2400, default_order=2 &end\n"
            "&run_control &end\n"
            "&twiss_output filename=twiss.sdds, matched=1, radiation_integrals=1, output_at_each_step=1 &end\n"
            "&rf_setup harmonic=80, total_voltage=1500000, set_for_each_step=1 &end\n"
            "&closed_orbit output=orbit.sdds, fixed_length=2 &end\n"
            "&moments_output filename=moments.sdds, matrix_output=matrices.sdds, matched=1,"
            " equilibrium=1, radiation=1, n_slices=20, output_at_each_step=1 &end\n"
            "&bunched_beam &end\n&track &end\n"
        )
        result = run(
            {"ring.lte": LATTICE, "run.ele": commands}, outputs=("moments.sdds", "matrices.sdds")
        )
        moments, matrices = result["moments.sdds"], result["matrices.sdds"]
        sigma = np.empty((6, 6))
        for i in range(6):
            for j in range(i, 6):
                value = moments[0, f"s{i + 1}"][0] ** 2 if i == j else moments[0, f"s{i + 1}{j + 1}"][0]
                sigma[i, j] = sigma[j, i] = value
        m = np.array([[matrices[0, f"R{i}{j}"][0] for j in range(1, 7)] for i in range(1, 7)])
        d = np.array([[matrices[0, f"D{i}{j}"][0] for j in range(1, 7)] for i in range(1, 7)])
        solved = solve_discrete_lyapunov(m, d)
        agreement = np.abs(solved - sigma).max() / np.abs(solved).max()
        print(
            f"start {line} control: SciPy equilibrium covariance vs Elegant's, max relative deviation {agreement:.2e}"
        )
        if agreement > 1e-9:
            sys.exit(
                "Elegant's equilibrium covariance disagrees with the Lyapunov solution: invalid comparison."
            )
        canonical = eigen_emittances(solved, block_diag(J, J, -J))
        same_sign = mode_emittances(m, solved, block_diag(J, J, J))
        flipped = mode_emittances(m, solved, block_diag(J, J, -J))
        # Ordered by size, the modes are (vertical ~0, horizontal, longitudinal) here.
        expected = [canonical[1], canonical[2]]
        observed = [float(moments["e1"][0]), float(moments["e3"][0])]
        print(
            f"start {line} e2 (m): reported={float(moments['e2'][0]):.3g} expected={canonical[0]:.3g}"
        )
        print(
            f"start {line} e1, e3 (m): observed=[{observed[0]:.10g}, {observed[1]:.10g}]"
            f" expected=[{expected[0]:.10g}, {expected[1]:.10g}]"
        )
        print(
            f"start {line} relative error e1, e3: {observed[0] / expected[0] - 1:+.4%}, {observed[1] / expected[1] - 1:+.4%}"
        )
        print(
            f"start {line} eigenvector normalization with diag(J, J, J): e1, e3 = [{same_sign[1]:.10g},"
            f" {same_sign[2]:.10g}]; with diag(J, J, -J): [{flipped[1]:.10g}, {flipped[2]:.10g}]"
        )
        defect |= not np.allclose(observed, expected, rtol=RTOL, atol=0)
    verdict(defect)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`claude-opus-5-5`</span> | high | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |
| Confirmed<br><small class="bb-scope">e1 and e3; e2 is zero in the confirming ring</small> | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-04</span> | master (`7514db36f764`), source with SDDS a3f730a14c49, CPU, serial |

