---
title: "ENV-01: Eigenemittance diagnostics corrupt small or repeated modes"
hide:
  - toc
---

# ENV-01 — Eigenemittance diagnostics corrupt small or repeated modes

| | |
|---|---|
| Code | [ImpactX](../index.md) · Envelope |
| Priority | P2 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX commit [`fa0a43b`](https://github.com/BLAST-ImpactX/impactx/commit/fa0a43b9bb07), committed 2026-10-28 |
| Issue | [[gh:https://github.com/BLAST-ImpactX/impactx/issues/1481]] |
| Discussed in | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1757]] |
| Source | [`EmittanceInvariants.cpp#L119-L134`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/diagnostics/EmittanceInvariants.cpp#L119-L134)<br>[`CovarianceMatrixMath.H#L59-L90`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/diagnostics/CovarianceMatrixMath.H#L59-L90) |

With eigenemittance diagnostics enabled, `beam_moments()` returns wrong eigenemittances, or three NaNs, when the beam's emittances differ widely or two are equal, for both particle beams and `Envelope`. In the source, the characteristic cubic is formed from differences of trace invariants, which loses the small modes, and `acos` is then evaluated without guarding its domain. For an uncoupled beam the documented eigenemittances equal the projected normalized rms emittances, which an independent Hermitian eigensolver reproduces to within 5×10⁻¹⁵ relative. At γ = 2, an uncoupled beam with normalized emittances 0.01, 0.02 and 1000 µm gives eigenemittances 0.05116, 0.05584 and 1000 µm, and 1, 1 and 1000 µm give three NaNs. A moderate spread (1, 2 and 3 µm) is correct, as are the projected emittances.

Run `python ENV-01.py`.

??? example "Reproducer: ENV-01.py"

    [Download ENV-01.py](ENV-01.py)

    ```python
    """ENV-01: Eigenemittance diagnostics corrupt small or repeated modes

    With eigenemittance diagnostics enabled, `beam_moments()` returns wrong eigenemittances, or three
    NaNs, when the beam's emittances differ widely or two are equal, for both particle beams and
    `Envelope`. In the source, the characteristic cubic is formed from differences of trace invariants,
    which loses the small modes, and `acos` is then evaluated without guarding its domain. For an
    uncoupled beam the documented eigenemittances equal the projected normalized rms emittances, which
    an independent Hermitian eigensolver reproduces to within 5×10⁻¹⁵ relative. At γ = 2, an uncoupled
    beam with normalized emittances 0.01, 0.02 and 1000 µm gives eigenemittances 0.05116, 0.05584 and
    1000 µm, and 1, 1 and 1000 µm give three NaNs. A moderate spread (1, 2 and 3 µm) is correct, as are
    the projected emittances.

    The script builds diagonal covariance matrices at γ = 2 for three sets of normalized emittances (1,
    2 and 3 µm as the control; 0.01, 0.02 and 1000 µm; 1, 1 and 1000 µm) and passes each to
    `Envelope.beam_moments()` and, as twelve equal-weight particles with exactly that covariance, to the
    particle `beam_moments()`. It compares the eigenemittances with the analytic values, which the
    eigenvalues of the Hermitian matrix i·√Σ·J·√Σ confirm (Σ the normalized covariance, J the symplectic
    form), and checks that the projected normalized emittances are exact. The control must pass in both
    interfaces and the other two sets must fail.
    """

    import os

    for variable in (
        "NPROC",
        "PJRT_NPROC",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import numpy as np
    from impactx import Envelope, ImpactX, Map6x6

    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.eigenemittances = True
    sim.init_grids()
    ref = sim.beam.ref
    ref.set_charge_qe(1).set_mass_MeV(1000).set_kin_energy_MeV(1000)
    bg = ref.beta_gamma
    print(f"gamma={ref.gamma:.17g} beta_gamma={bg:.17g}")
    j = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])


    def compare(label, moments, expected):
        projected = np.array([moments[f"emittance_{axis}n"] for axis in "xyt"])
        eigen = np.sort([moments[f"emittance_{i}"] for i in (1, 2, 3)])
        np.testing.assert_allclose(np.sort(projected), expected, rtol=2e-14, atol=0)
        good = np.all(np.isfinite(eigen)) and np.max(abs(eigen / expected - 1)) < 1e-11
        print(
            f"{label}: expected={expected.tolist()} projected={projected.tolist()} eigen={eigen.tolist()} valid={good}"
        )
        return good


    try:
        for emittances in ([1e-6, 2e-6, 3e-6], [1e-8, 2e-8, 1e-3], [1e-6, 1e-6, 1e-3]):
            expected = np.array(emittances)
            # Covariances in (x,px,y,py,t,pt); p's are normalized to reference P0.
            diagonal = np.repeat(expected, 2)
            diagonal[1::2] /= bg**2
            matrix = Map6x6()
            raw = matrix.to_numpy()
            raw[:] = np.diag(diagonal)
            normalized = raw.copy()
            normalized[1::2, :] *= bg
            normalized[:, 1::2] *= bg
            assert np.linalg.eigvalsh(normalized).min() > 0
            # For SPD Sigma, i*sqrt(Sigma)*J*sqrt(Sigma) is Hermitian and has
            # eigenvalues +/- epsilon.  This avoids a characteristic polynomial.
            square_root = np.diag(np.sqrt(np.diag(normalized)))
            oracle = np.linalg.eigvalsh(1j * square_root @ j @ square_root)[3:]
            np.testing.assert_allclose(oracle, expected, rtol=5e-15, atol=0)
            print(
                f"SPD condition_number={np.linalg.cond(normalized):.8g} hermitian_oracle={oracle.tolist()}"
            )
            env_good = compare("envelope", Envelope(matrix, 0.0).beam_moments(ref), expected)

            # Twelve equal-weight particles with exactly diagonal sample covariance:
            # each coordinate has two entries +/-sqrt(6*variance), all others zero.
            points = np.concatenate((np.diag(np.sqrt(6 * diagonal)), -np.diag(np.sqrt(6 * diagonal))))
            sim.beam.clear_particles()
            sim.beam.add_n_particles(
                x=points[:, 0].copy(),
                px=points[:, 1].copy(),
                y=points[:, 2].copy(),
                py=points[:, 3].copy(),
                t=points[:, 4].copy(),
                pt=points[:, 5].copy(),
                qm=ref.qm_ratio_SI,
                bunch_charge=1e-15,
            )
            particles_good = compare("particles", sim.beam.beam_moments(), expected)
            if emittances[-1] == 3e-6:
                assert env_good and particles_good
            else:
                assert not env_good and not particles_good
        print("DEFECT REPRODUCED")
    finally:
        sim.finalize()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

