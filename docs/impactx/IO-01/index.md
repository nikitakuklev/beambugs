---
title: "IO-01: A tracer can erase weighted beam size and emittance"
hide:
  - toc
---

# IO-01 — A tracer can erase weighted beam size and emittance

| | |
|---|---|
| Code | [ImpactX](../index.md) · Particles and I/O |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`, head build) |
| Discussed in | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1757]] |
| Source | [`ReducedBeamCharacteristics.cpp#L197-L203`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/diagnostics/ReducedBeamCharacteristics.cpp#L197-L203)<br>[`L335`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/diagnostics/ReducedBeamCharacteristics.cpp#L335) |

`beam_moments()` can report zero beam size and emittance for a finite-emittance beam when a distant particle with zero or tiny weight, such as a tracer, is stored first. The moments are accumulated about the first stored particle, so the shifted second moment and squared mean nearly cancel and the core variance is lost. Weighted standard deviations and emittances must not depend on particle order or zero-weight particles; a two-pass weighted covariance gives the correct values. For a core with σx = 10⁻⁹ m and emittance εx = 10⁻¹² m and a tracer at x = 1 m stored first, with weight 0 or 10⁻²⁴, both are reported as 0; stored last, they are correct.

Run `python IO-01.py`.

??? example "Reproducer: IO-01.py"

    [Download IO-01.py](IO-01.py)

    ```python
    """IO-01: A tracer can erase weighted beam size and emittance

    `beam_moments()` can report zero beam size and emittance for a finite-emittance beam when a distant
    particle with zero or tiny weight, such as a tracer, is stored first. The moments are accumulated
    about the first stored particle, so the shifted second moment and squared mean nearly cancel and the
    core variance is lost. Weighted standard deviations and emittances must not depend on particle order
    or zero-weight particles; a two-pass weighted covariance gives the correct values. For a core with
    σx = 10⁻⁹ m and emittance εx = 10⁻¹² m and a tracer at x = 1 m stored first, with weight 0 or 10⁻²⁴,
    both are reported as 0; stored last, they are correct.

    The script builds a four-particle core at all sign combinations of x = ±σ and px = ±10⁻³ plus a
    tracer at x = 1 m, for σ = 10⁻⁵, 10⁻⁷ and 10⁻⁹ m and tracer weights 0 and 10⁻²⁴, and calls
    `beam_moments()` with the tracer stored first and last. It compares σx and εx with a two-pass
    weighted covariance: the tracer-last order must agree, and at σ = 10⁻⁹ m the tracer-first order
    reports zero for both.
    """

    import os

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"

    import numpy as np
    from impactx import ImpactX

    COORDS = (
        "position_x",
        "momentum_x",
        "position_y",
        "momentum_y",
        "position_t",
        "momentum_t",
    )


    class Oracle:
        def __init__(self):
            self.sim = ImpactX()
            self.sim.particle_shape = 1
            self.sim.space_charge = False
            self.sim.diagnostics = False
            self.sim.slice_step_diagnostics = False
            self.sim.verbose = 0
            self.sim.init_grids()
            self.ref = self.sim.beam.ref

        def load(self, u, weights=None, species="proton", gamma=2.0, anomaly=None, spin=None):
            self.sim.beam.clear_particles()
            self.ref.reset()
            self.ref.set_species(species)
            self.ref.set_kin_energy_MeV((gamma - 1) * self.ref.mass_MeV)
            if anomaly is not None:
                self.ref.set_gyromagnetic_anomaly(anomaly)
            self.sim.spin = spin is not None
            u = np.atleast_2d(np.asarray(u, dtype=float))
            kwargs = dict(zip(("x", "px", "y", "py", "t", "pt"), u.T.copy()))
            kwargs["qm"] = self.ref.qm_ratio_SI
            kwargs["w"] = np.ones(len(u)) if weights is None else np.asarray(weights, dtype=float)
            if spin is not None:
                kwargs.update({k: np.full(len(u), v) for k, v in zip(("sx", "sy", "sz"), spin)})
            self.sim.beam.add_n_particles(**kwargs)

        def push(self, element):
            self.sim.lattice.clear()
            self.sim.lattice.append(element)
            self.sim.track_particles()
            return self.sim.beam.to_df(local=True)

        def close(self):
            self.sim.finalize()


    o = Oracle()
    try:
        for sigma in (1e-5, 1e-7, 1e-9):
            for tracer_weight in (0.0, 1e-24):
                u = np.zeros((5, 6))
                u[0, 0] = 1
                u[1:, 0] = sigma * np.array([-1, -1, 1, 1])
                u[1:, 1] = 1e-3 * np.array([-1, 1, -1, 1])
                weights = np.array([tracer_weight, 1, 1, 1, 1])
                # Center on the weighted mean directly. No subtraction of O(1) second moments.
                center = np.sum(weights[:, None] * u, axis=0) / np.sum(weights)
                deviations = u - center
                cov = (deviations.T * weights) @ deviations / weights.sum()
                expected_sigma = np.sqrt(cov[0, 0])
                expected_emit = np.sqrt(cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2)
                for label, order in (
                    ("tracer_first", [0, 1, 2, 3, 4]),
                    ("tracer_last", [1, 2, 3, 4, 0]),
                ):
                    o.load(u[order], weights[order])
                    m = o.sim.beam.beam_moments()
                    print(
                        f"sigma={sigma:g} tracer_weight={tracer_weight:g} {label}: "
                        f"sigma_x={m['sigma_x']:.15g} expected={expected_sigma:.15g} "
                        f"emit_x={m['emittance_x']:.15g} expected={expected_emit:.15g}",
                        flush=True,
                    )
                    if label == "tracer_last":
                        np.testing.assert_allclose(m["sigma_x"], expected_sigma, rtol=2e-14, atol=0)
                        np.testing.assert_allclose(m["emittance_x"], expected_emit, rtol=2e-14, atol=0)
                    if sigma == 1e-9 and label == "tracer_first":
                        assert m["sigma_x"] == 0 and m["emittance_x"] == 0
        print("DEFECT REPRODUCED")
    finally:
        o.close()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

