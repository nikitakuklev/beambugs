---
title: "ENV-02: Nonlinear DipEdge optics omit K5 fringe focusing"
hide:
  - toc
---

# ENV-02 — Nonlinear `DipEdge` optics omit `K5` fringe focusing

| | |
|---|---|
| Code | [ImpactX](../index.md) · Envelope |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on development (`fa0a43b9bb07`), source, CPU, double, serial |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`DipEdge.H#L433-L467`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/DipEdge.H#L433-L467)<br>[`DipEdge.H#L210-L287`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/DipEdge.H#L210-L287) |

For a `DipEdge` with the nonlinear fringe model, the linear transfer matrix that ImpactX reports omits the vertical focusing from the fringe-field integral `K5`, which the particle map includes. The reported matrix must equal the small-amplitude derivative of the particle map, whose vertical focusing changes in proportion to `sin(psi)·g·K5/cos³(psi)`. For a proton at 1000 MeV kinetic energy with `psi = 0.2 rad`, `rc = 1 m`, `g = 0.1 m` and `K1 = 0`, raising `K5` from 0 to 0.1 changes the particle map's `R43 = ∂py/∂y` by 0.002110 m⁻¹ while the reported matrix stays unchanged. Envelope tracking uses the reported matrix.

## Reproducer

??? example "ENV-02.py"

    [Download ENV-02.py](ENV-02.py)

    ```python
    """ENV-02: Nonlinear DipEdge optics omit K5 fringe focusing at K1=0."""

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

    from importlib.metadata import version
    from math import cos, pi, sin, sqrt, tan

    import numpy as np
    from impactx import ImpactX, elements, push

    print("impactx-noacc", version("impactx-noacc"))
    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.verbose = 0
    sim.init_grids()


    def measure(k1, gap, radius, model):
        sim.beam.clear_particles()
        ref = sim.beam.ref
        ref.reset()
        ref.set_species("proton").set_kin_energy_MeV(1000)
        # At psi=0, x=px=py=pt=0 and K4=K5=K6=0, the nonlinear map reduces to
        # py' = [g/rc^2*(K2+K1/2)] y - [2*K3/(3*rc^2*g)] y^3.
        # K0 only shifts x; keep its physical default. Retain default K2=1,K3=1/6.
        h = 1e-4
        ys = np.array([h, -h, h / 2, -h / 2])
        z = np.zeros(4)
        sim.beam.add_n_particles(
            x=z.copy(),
            px=z.copy(),
            y=ys.copy(),
            py=z.copy(),
            t=z.copy(),
            pt=z.copy(),
            qm=ref.qm_ratio_SI,
            w=np.ones(4),
        )
        edge = elements.DipEdge(psi=0, rc=radius, g=gap, K1=k1, model=model)
        reported = elements.KnownElementsList([edge]).transfer_map(ref).to_numpy().copy()
        push(sim.beam, edge)
        actual_py = sim.beam.to_df(local=True)["momentum_y"].to_numpy()
        linear = gap / radius**2 * (1 + k1 / 2) if model == "nonlinear" else gap / radius**2
        cubic = -2 * (1 / 6) / (3 * radius**2 * gap) if model == "nonlinear" else 0
        np.testing.assert_allclose(actual_py, linear * ys + cubic * ys**3, rtol=4e-15, atol=1e-22)
        d_h = (actual_py[0] - actual_py[1]) / (2 * h)
        d_half_h = (actual_py[2] - actual_py[3]) / h
        # Richardson extrapolation cancels the known cubic term exactly.
        derivative = (4 * d_half_h - d_h) / 3
        np.testing.assert_allclose(derivative, linear, rtol=0, atol=8e-17)
        if model == "linear" or k1 == 0:
            np.testing.assert_allclose(reported[3, 2], derivative, rtol=0, atol=8e-17)
        else:
            np.testing.assert_allclose(
                derivative - reported[3, 2], gap * k1 / (2 * radius**2), rtol=0, atol=8e-17
            )
        print(
            f"{model} K1={k1:g} gap={gap:g} rc={radius:g}: particle_R43={derivative:.16g} analytic_R43={linear:.16g} reported_R43={reported[3, 2]:.16g} missing={derivative - reported[3, 2]:.16g}"
        )


    def curved_field(k5):
        """Test K5 at K1=0, respecting Hwang--Lee's design-angle convention."""
        sim.beam.clear_particles()
        ref = sim.beam.ref
        ref.reset()
        ref.set_species("proton").set_kin_energy_MeV(1000)
        psi, gap, radius, scale = 0.2, 0.1, 1.0, 1.0
        edge = elements.DipEdge(psi=psi, rc=radius, g=gap, R=scale, K1=0, K5=k5, model="nonlinear")
        h = 2e-5
        ys = np.array([h, -h, h / 2, -h / 2])
        z = np.zeros(4)
        sim.beam.add_n_particles(
            x=z.copy(),
            px=z.copy(),
            y=ys.copy(),
            py=z.copy(),
            t=z.copy(),
            pt=z.copy(),
            qm=ref.qm_ratio_SI,
            w=np.ones(4),
        )
        reported = elements.KnownElementsList([edge]).transfer_map(ref).to_numpy()[3, 2]
        push(sim.beam, edge)
        py = sim.beam.to_df(local=True)["momentum_y"].to_numpy()
        derivative = (4 * (py[2] - py[3]) / h - (py[0] - py[1]) / (2 * h)) / 3
        c5 = tan(psi) / (2 * radius)
        c7 = gap * (1 + sin(psi) ** 2) / (2 * radius**2 * cos(psi) ** 3)
        c14 = sin(psi) * gap * k5 / (2 * radius * scale * cos(psi) ** 3)
        c10 = tan(psi) ** 2 / (2 * radius)
        c11 = 1 / (2 * radius)
        c2 = gap**2 * (pi**2 / 6) * sin(psi) / (2 * radius**2 * cos(psi) ** 3)
        c3 = gap**2 * (pi**2 / 6) / (radius * cos(psi) ** 2)
        expected = 2 * (c14 - c5 + c7) + 2 * (c10 + c11) * (c2 + c3 * c5)
        np.testing.assert_allclose(derivative, expected, rtol=0, atol=2e-16)
        print(
            f"nonlinear K1=0 K5={k5:g} psi={psi:g}: particle_R43={derivative:.16g} analytic_R43={expected:.16g} reported_R43={reported:.16g}"
        )
        return derivative, reported


    try:
        measure(0, 0.1, 1, "nonlinear")
        measure(0, 0.1, 1, "linear")
        # A realizable on-axis logistic field has K1=0, K0=pi^2/6, K2=1, K3=1/6.
        # Independently choose its horizontal second derivative as
        # (d^2 B_y/dx^2)/B0 = A*s/(g^2*R)*exp[-s^2/(2*g^2)]. It vanishes at both
        # boundaries; its zeroth and second moments vanish, while its first moment
        # sets K5=A*sqrt(2*pi), with K4=K6=0. This fits the cited paper's general
        # midplane-symmetric scalar-potential model, without the optional assumption
        # that field contours have a fixed curvature.
        a = 0.1 / sqrt(2 * pi)
        nodes, weights = np.polynomial.hermite.hermgauss(48)
        k4 = 4 * a * np.sum(weights * nodes**3)
        k5 = 2 * sqrt(2) * a * np.sum(weights * nodes**2)
        k6 = 2 * a * np.sum(weights * nodes)
        np.testing.assert_allclose([k4, k5, k6], [0, 0.1, 0], rtol=0, atol=3e-16)
        print(f"physical field moments K1=0 K4={k4:.3g} K5={k5:.16g} K6={k6:.3g}")
        d0, r0 = curved_field(0)
        d1, r1 = curved_field(0.1)
        expected_change = sin(0.2) * 0.1 * 0.1 / cos(0.2) ** 3
        np.testing.assert_allclose(d1 - d0, expected_change, rtol=0, atol=2e-16)
        assert r1 == r0
        print(
            f"K5-induced change: particle={d1 - d0:.16g} expected={expected_change:.16g} reported={r1 - r0:.16g}"
        )
    finally:
        sim.finalize()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

