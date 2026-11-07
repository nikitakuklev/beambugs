---
title: "API-05: Survey reverses normalized-quadrupole focusing for negative charge"
hide:
  - toc
---

# API-05 — Survey reverses normalized-quadrupole focusing for negative charge

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX development (`fa0a43b`, head build) |
| Source | [`Survey.py#L85-L86`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/python/impactx/plot/Survey.py#L85-L86) |

`plot_survey()` draws each quadrupole above or below the axis according to the sign of `k`·charge, but `Quad.k` and `ChrQuad(unit=0).k` are already normalized by the signed rigidity, so a positive `k` focuses horizontally for either charge. The plot's docstring defines a positive height as horizontal focusing, and particle tracking and the analytic transfer matrix confirm the focusing direction for both charges. A 0.5 m `Quad` with `k` = 1 m⁻² focuses a 1000 MeV electron exactly as it does a proton (matrix element R₂₁ = −0.4794 m⁻¹), yet it is drawn with height −0.8 instead of +0.8. For electrons and other negative species every `Quad` and normalized `ChrQuad` is drawn reversed; tracking is correct, and `ChrQuad` with a physical gradient (`unit=1`) is drawn correctly.

Run `python API-05.py`.

??? example "Reproducer: API-05.py"

    [Download API-05.py](API-05.py)

    ```python
    """API-05: Survey reverses normalized-quadrupole focusing for negative charge

    `plot_survey()` draws each quadrupole above or below the axis according to the sign of `k`·charge,
    but `Quad.k` and `ChrQuad(unit=0).k` are already normalized by the signed rigidity, so a positive
    `k` focuses horizontally for either charge. The plot's docstring defines a positive height as
    horizontal focusing, and particle tracking and the analytic transfer matrix confirm the focusing
    direction for both charges. A 0.5 m `Quad` with `k` = 1 m⁻² focuses a 1000 MeV electron exactly as
    it does a proton (matrix element R₂₁ = −0.4794 m⁻¹), yet it is drawn with height −0.8 instead of
    +0.8. For electrons and other negative species every `Quad` and normalized `ChrQuad` is drawn
    reversed; tracking is correct, and `ChrQuad` with a physical gradient (`unit=1`) is drawn correctly.

    The script builds 0.5 m `Quad`, `ChrQuad(unit=0)` and `ChrQuad(unit=1)` elements with normalized
    strength k = ±1 m⁻² (the `unit=1` gradient scaled by the rigidity) for 1000 MeV protons and
    electrons, and checks R₂₁ from the reported linear map and from a pushed particle at x = 1 µm
    against the analytic −√k sin(√k L) or √(−k) sinh(√(−k) L). It then reads each rectangle's height
    from `plot_survey()`, drawn with Matplotlib's noninteractive backend, and expects +0.8 for k > 0 and
    −0.8 for k < 0. Protons and the `unit=1` elements are the controls; the four normalized electron
    cases are drawn reversed.
    """

    import os
    from pathlib import Path
    from tempfile import TemporaryDirectory

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"
    plot_config = TemporaryDirectory(prefix=".API-05-mpl-", dir=Path(__file__).resolve().parent)
    os.environ["MPLCONFIGDIR"] = plot_config.name

    from math import sin, sinh

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from impactx import ImpactX, elements, push
    from matplotlib.patches import Rectangle

    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.verbose = 0
    sim.init_grids()
    mismatches = 0
    try:
        for species in ("proton", "electron"):
            for norm_k in (1.0, -1.0):
                for kind in ("Quad", "ChrQuad_normalized", "ChrQuad_T_per_m"):
                    sim.beam.clear_particles()
                    ref = sim.beam.ref
                    ref.reset()
                    ref.set_species(species).set_kin_energy_MeV(1000)
                    if kind == "Quad":
                        el = elements.Quad(ds=0.5, k=norm_k)
                    elif kind == "ChrQuad_normalized":
                        el = elements.ChrQuad(ds=0.5, k=norm_k, unit=0)
                    else:
                        el = elements.ChrQuad(ds=0.5, k=norm_k * ref.rigidity_Tm, unit=1)
                    lattice = elements.KnownElementsList([el])
                    exact_R21 = -sin(0.5) if norm_k > 0 else sinh(0.5)
                    reported_R21 = lattice.transfer_map(ref).to_numpy()[1, 0]
                    np.testing.assert_allclose(reported_R21, exact_R21, rtol=0, atol=2e-15)
                    z = np.zeros(1)
                    sim.beam.add_n_particles(
                        x=np.array([1e-6]),
                        px=z.copy(),
                        y=z.copy(),
                        py=z.copy(),
                        t=z.copy(),
                        pt=z.copy(),
                        qm=ref.qm_ratio_SI,
                        w=np.ones(1),
                    )
                    push(sim.beam, el)
                    actual_R21 = sim.beam.to_df(local=True)["momentum_x"].to_numpy()[0] / 1e-6
                    np.testing.assert_allclose(actual_R21, exact_R21, rtol=0, atol=2e-15)
                    fig, ax = plt.subplots()
                    try:
                        lattice.plot_survey(ref=ref, ax=ax, legend=False)
                        rectangle = ax.patches[0]
                        assert isinstance(rectangle, Rectangle)
                        height = rectangle.get_height()
                    finally:
                        plt.close(fig)
                    expected_height = 0.8 if norm_k > 0 else -0.8
                    is_defect = species == "electron" and kind != "ChrQuad_T_per_m"
                    assert (height != expected_height) == is_defect
                    mismatches += is_defect
                    print(
                        f"{species} {kind} normalized_k={norm_k:g}: particle_R21={actual_R21:.15g} analytic_R21={exact_R21:.15g} height={height:+g} expected={expected_height:+g}"
                    )
    finally:
        sim.finalize()
        plot_config.cleanup()
    assert mismatches == 4
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

