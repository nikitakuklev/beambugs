---
title: "LAT-17: The manual calls a momentum spread an energy spread"
hide:
  - toc
---

# LAT-17 — The manual calls a momentum spread an energy spread

| | |
|---|---|
| Code | [Bmad](../index.md) · Lattices, particles and I/O |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-05 on Bmad 20261031-0 (`c4c629f`) |
| Source | [init_random_distribution longitudinal transform, bmad/multiparticle/beam_utils.f90:675–706](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/multiparticle/beam_utils.f90#L675-L706)<br>[Bmad manual, Beam_Init_Struct Structure (`s:beam.init.struct`), bmad/doc/beam-init.tex:247–253](https://github.com/bmad-sim/bmad-ecosystem/blob/c4c629fc9071d3a56cb1ba5bc714baba65e9c5d3/bmad/doc/beam-init.tex#L247-L253) |

Initializing an ordinary Gaussian bunch with sig_pz sets σδ, the RMS fractional momentum spread, although the manual calls it fractional energy spread. At finite speed these differ: σE/E₀ = β₀² σδ to first order about the reference particle. Here E₀ is reference energy, σE is RMS energy spread, β₀ is reference speed divided by light speed, and δ is momentum deviation divided by reference momentum. For an electron reference momentum of 1 MeV/c and sig_pz = 0.002, the generated σδ is 0.002, while the relative energy spread is smaller. The longitudinal covariance and mass-shell relation establish the momentum convention.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _bmad.py](../helpers/_bmad.py) · [View source](../helpers/index.md#_bmad)
- [Download _s3_12_particle_contracts.py](../helpers/_s3_12_particle_contracts.py) · [View source](../helpers/index.md#_s3_12_particle_contracts)

??? example "Reproducer: LAT-17.py"

    [Download LAT-17.py](LAT-17.py)

    ```python
    """LAT-17: The manual calls a momentum spread an energy spread

    Initializing an ordinary Gaussian bunch with sig_pz sets σδ, the RMS fractional momentum spread, although the manual calls it fractional energy spread. At finite speed these differ: σE/E₀ = β₀² σδ to first order about the reference particle. Here E₀ is reference energy, σE is RMS energy spread, β₀ is reference speed divided by light speed, and δ is momentum deviation divided by reference momentum. For an electron reference momentum of 1 MeV/c and sig_pz = 0.002, the generated σδ is 0.002, while the relative energy spread is smaller. The longitudinal covariance and mass-shell relation establish the momentum convention.

    The script initializes the deterministic Gaussian case with seed 314159 and recomputes charge-weighted moments from emitted particles. It compares the longitudinal covariance with the requested momentum covariance and reconstructs energies from the mass shell. BMAD_SOURCE supplies the conflicting manual sentence.
    """

    import io

    import numpy as np
    from _bmad import compare, fortran, verdict
    from _s3_12_particle_contracts import INIT, document_claim, rows

    result = fortran(INIT, ["pseudo"], outputs=["particles.dat"])
    data = np.loadtxt(io.BytesIO(result.outputs["particles.dat"]))
    if data.shape != (2048, 9) or not np.isfinite(data).all() or not (data[:, 6] > 0).all():
        raise RuntimeError("Expected 2048 finite particles with positive charge weights")
    mass = rows(result.log)["init"][4]
    w = data[:, 6].astype(np.longdouble)
    w /= w.sum()
    center = w @ data[:, :6]
    d = data[:, :6] - center
    cov = (d.T * w) @ d
    # A momentum chirp d(delta)/dz gives Cov(z,delta)=chirp*Var(z).
    sig_z, sig_delta, chirp = 0.004, 0.002, 0.2
    expected = np.array([[sig_z**2, chirp * sig_z**2], [chirp * sig_z**2, sig_delta**2]])
    if compare(
        "covariance (z,delta), momentum/chirp expectation",
        cov[4:, 4:],
        expected,
        atol=2e-18,
        rtol=2e-12,
    ):
        raise RuntimeError("Momentum covariance control failed")
    pc = 1e6 * (1 + data[:, 5])
    energy = np.hypot(pc, mass)
    np.testing.assert_allclose(data[:, 7], pc / energy, rtol=3e-15)
    # E^2=(pc)^2+(mc^2)^2 gives dE/E=beta^2 dp/p at the reference particle.
    e_rel = energy / np.hypot(1e6, mass)
    sigma_e = np.sqrt(w @ (e_rel - w @ e_rel) ** 2)
    beta0_sq = 1e12 / (1e12 + mass**2)
    print(
        f"sigma_delta={float(np.sqrt(cov[5, 5])):.16g}; sigma_E/E0={float(sigma_e):.16g}; beta0^2={beta0_sq:.16g}; first-order at delta=0={beta0_sq * 0.002:.16g}"
    )
    verdict(
        document_claim("bmad/doc/beam-init.tex", r"\vn{%sig_pz} is the fractional energy spread dE/E.")
    )

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-20 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6-astra` | xhigh | 2026-10-21 | 20261020-0 (`19b8afb46a40`), source, production, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-05 | 20261031-0 (`c4c629fc9071`), source, production, GNU Fortran 15.3.0, CPU, double, serial, OpenMP=1 |

