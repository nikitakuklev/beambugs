---
title: "ENV-03: Phase-space legends label geometric emittance as normalized"
hide:
  - toc
---

# ENV-03 — Phase-space legends label geometric emittance as normalized

| | |
|---|---|
| Code | [ImpactX](../index.md) · Envelope |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-28 on ImpactX 26.10 (`615921c`); not reproduced on ImpactX development (`f9c2a82`), 2026-11-05 |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1762]], fix |
| Source | [`ImpactXParticleContainer.py#L257-L285`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/extensions/ImpactXParticleContainer.py#L257-L285) |

The phase-space plot from `plot_phasespace()` labels its legends with the normalized-emittance symbols εₙ,x, εₙ,y and εₙ,t but displays the geometric emittances. ImpactX documents the normalized emittances as the diagnostic keys with an `n` suffix, equal to β₀γ₀ times the geometric values, with β₀ the reference speed over c and γ₀ the reference Lorentz factor. For a proton beam at γ₀ = 7, the x legend shows 168.7 nm while the normalized emittance is 1168 nm, a factor β₀γ₀ = 6.928. The numerical diagnostics are correct; only the plot annotation is wrong, by the factor β₀γ₀.

??? example "Reproducer: ENV-03.py"

    [Download ENV-03.py](ENV-03.py)

    ```python
    """ENV-03: Phase-space legends label geometric emittance as normalized

    The phase-space plot from `plot_phasespace()` labels its legends with the normalized-emittance
    symbols εₙ,x, εₙ,y and εₙ,t but displays the geometric emittances. ImpactX documents the normalized
    emittances as the diagnostic keys with an `n` suffix, equal to β₀γ₀ times the geometric values, with
    β₀ the reference speed over c and γ₀ the reference Lorentz factor. For a proton beam at γ₀ = 7, the
    x legend shows 168.7 nm while the normalized emittance is 1168 nm, a factor β₀γ₀ = 6.928. The
    numerical diagnostics are correct; only the plot annotation is wrong, by the factor β₀γ₀.

    The script builds a deterministic weighted product beam at γ₀ = 7 and checks its geometric and normalized emittances against analytic values. It compares the installed plot legends with the normalized-emittance diagnostics, then repeats with each particle split into two half-weight copies. The verdict comes from the displayed values; particle splitting must preserve the diagnostics.
    """

    import os

    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[key] = "1"
    from pathlib import Path
    from tempfile import TemporaryDirectory

    plot_config = TemporaryDirectory(prefix=".ENV-03-mpl-", dir=Path(__file__).resolve().parent)
    os.environ["MPLCONFIGDIR"] = plot_config.name
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    from importlib import import_module
    from importlib.metadata import version
    from itertools import product

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from impactx import ImpactX
    from quantiphy import Quantity

    module = import_module("impactx.extensions.ImpactXParticleContainer")
    bad = False


    # Each independent +/- coordinate has probability ratio 1:r. Analytic stddev:
    # sigma = 2 a sqrt(r)/(1+r). All coordinate covariances are zero.
    ratios = np.array([2.0, 5.0, 3.0, 7.0, 11.0, 13.0])
    amplitudes = np.array([0.8e-3, 0.3e-3, 0.6e-3, 0.2e-3, 1.5e-3, 4e-3])
    bits = np.array(list(product([0, 1], repeat=6)))
    u = (2 * bits - 1) * amplitudes
    weights = 1e5 * np.prod(ratios**bits, axis=1)
    sigmas = 2 * amplitudes * np.sqrt(ratios) / (1 + ratios)
    emittances = sigmas[::2] * sigmas[1::2]
    gamma = 7.0
    bg = np.sqrt(gamma**2 - 1)

    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.verbose = 0
    sim.init_grids()

    try:
        print(
            "impactx-noacc",
            version("impactx-noacc"),
            "numpy",
            version("numpy"),
            "matplotlib",
            version("matplotlib"),
        )
        print("Analytic product beam: 64 vertices; splitting each macroparticle preserves all physics")
        plotters = [("installed plotter", module.ix_pc_plot_mpl_phasespace)]
        for split in [1, 2]:
            sim.beam.clear_particles()
            ref = sim.beam.ref
            ref.reset()
            ref.set_species("proton").set_kin_energy_MeV((gamma - 1) * ref.mass_MeV)
            samples = np.repeat(u, split, axis=0)
            sample_weights = np.repeat(weights / split, split)
            sim.beam.add_n_particles(
                **dict(zip(("x", "px", "y", "py", "t", "pt"), samples.T.copy())),
                qm=ref.qm_ratio_SI,
                w=sample_weights,
            )
            moments = sim.beam.beam_moments()
            charge = ref.charge * 1e5 * np.prod(1 + ratios)
            np.testing.assert_allclose(moments["charge_C"], charge, rtol=3e-15, atol=0)
            for j, plane in enumerate("xyt"):
                np.testing.assert_allclose(
                    moments["emittance_" + plane], emittances[j], rtol=3e-14, atol=0
                )
                np.testing.assert_allclose(
                    moments["emittance_" + plane + "n"], emittances[j] * bg, rtol=3e-14, atol=0
                )
            for label, plotter in plotters:
                fig = plotter(sim.beam, num_bins=2)
                try:
                    panels = [axis for axis in fig.axes if axis.images]
                    assert len(panels) == 3
                    print(f"--- {label}; split={split}; charge={charge:.15g} C ---")
                    for j, (plane, panel) in enumerate(zip("xyt", panels)):
                        legend = panel.get_legend().get_title().get_text().splitlines()[0]
                        expected = (
                            rf"$\epsilon_{{n,{plane}}}=$"
                            + f"{Quantity(moments['emittance_' + plane + 'n'], 'm'):.3}"
                        )
                        bad |= legend != expected
                        print(
                            f"{plane}: legend={legend!r}; analytic geometric={emittances[j]:.15g} m; normalized={bg * emittances[j]:.15g} m"
                        )
                finally:
                    plt.close(fig)
    finally:
        sim.finalize()
        plot_config.cleanup()
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

