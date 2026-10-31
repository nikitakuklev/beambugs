---
title: "IO-03: Longitudinal plot scale and units disagree by 1000"
hide:
  - toc
---

# IO-03 — Longitudinal plot scale and units disagree by 1000

| | |
|---|---|
| Code | [ImpactX](../index.md) · Particles and I/O |
| Priority | P5 |
| Severity | Documentation or example mismatch |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`) |
| Source | [`ImpactXParticleContainer.py#L83-L95`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/extensions/ImpactXParticleContainer.py#L83-L95)<br>[`L117-L125`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/extensions/ImpactXParticleContainer.py#L117-L125)<br>[`L252`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/extensions/ImpactXParticleContainer.py#L252) |

The phase-space plot from `plot_phasespace()` multiplies the longitudinal momentum pt, the energy deviation −ΔE/(P₀c) with P₀ the reference momentum, by 1000, the factor it uses to convert transverse momenta to mrad, while the axis label states units of p₀c with no scale factor. ImpactX documents pt as dimensionless, so a per-mille display must state its factor. A beam with pt = ±0.004 is plotted at ±4. Tracking and stored coordinates are unaffected; values read from the plot are 1000 times too large.

??? example "Reproducer: IO-03.py"

    [Download IO-03.py](IO-03.py)

    ```python
    """IO-03: Longitudinal plot scale and units disagree by 1000

    The phase-space plot from `plot_phasespace()` multiplies the longitudinal momentum pt, the energy
    deviation −ΔE/(P₀c) with P₀ the reference momentum, by 1000, the factor it uses to convert
    transverse momenta to mrad, while the axis label states units of p₀c with no scale factor. ImpactX
    documents pt as dimensionless, so a per-mille display must state its factor. A beam with
    pt = ±0.004 is plotted at ±4. Tracking and stored coordinates are unaffected; values read from the
    plot are 1000 times too large.

    The script builds a deterministic beam of 64 weighted particles at the vertices of a box in (x, px,
    y, py, t, pt), with pt = ±0.004, for a γ = 7 proton reference, and draws the plot with the installed
    `plot_phasespace()` function and the unmodified function from the ImpactX development branch. It
    reads the longitudinal panel's image extent and axis label and checks that the extent is ±4 while
    the label carries no scale factor; splitting each particle into two half-weight copies gives the
    same result. Set IMPACTX_SOURCE to a clone of the ImpactX repository.
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

    plot_config = TemporaryDirectory(prefix=".IO-03-mpl-", dir=Path(__file__).resolve().parent)
    os.environ["MPLCONFIGDIR"] = plot_config.name
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

    import ast
    import inspect
    import subprocess
    from importlib import import_module
    from importlib.metadata import version
    from itertools import product

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from impactx import ImpactX
    from impactx.impactx_pybind import CoordSystem

    REPO = os.environ["IMPACTX_SOURCE"]
    PATH = "src/python/impactx/extensions/ImpactXParticleContainer.py"
    module = import_module("impactx.extensions.ImpactXParticleContainer")
    pinned = subprocess.check_output(["git", "-C", REPO, "show", f"615921c6867c:{PATH}"], text=True)
    assert inspect.getsource(module) == pinned


    def development_plot():
        source = subprocess.check_output(
            ["git", "-C", REPO, "show", f"b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943:{PATH}"], text=True
        )
        functions: list[ast.stmt] = [
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "ix_pc_plot_mpl_phasespace"
        ]
        assert len(functions) == 1
        namespace = {"CoordSystem": CoordSystem}
        exec(  # noqa: S102 -- execute the inspected upstream function without editing its body
            compile(
                ast.Module(body=functions, type_ignores=[]),
                "<unmodified development plotting function>",
                "exec",
            ),
            namespace,
        )
        return namespace["ix_pc_plot_mpl_phasespace"]


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
        plotters = [
            ("pinned 26.10", module.ix_pc_plot_mpl_phasespace),
            ("development function / 26.10 core", development_plot()),
        ]
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
                    longitudinal = panels[2]
                    extent = longitudinal.images[0].get_extent()[2:]
                    np.testing.assert_allclose(extent, [-4.0, 4.0], rtol=0, atol=1e-15)
                    assert longitudinal.get_ylabel() == r"$\Delta p_t$ [$p_0\cdot c$]"
                    print(
                        f"pt extent={list(extent)}; expected dimensionless extent=[-0.004, 0.004]; label={longitudinal.get_ylabel()!r}"
                    )
                finally:
                    plt.close(fig)
    finally:
        sim.finalize()
        plot_config.cleanup()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

