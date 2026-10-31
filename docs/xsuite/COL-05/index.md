---
title: "COL-05: Strong-strong 2D beam-beam moments include lost particles"
hide:
  - toc
---

# COL-05 — Strong-strong 2D beam-beam moments include lost particles

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P2 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xfields main (`223b4fa`) |
| Source | [`BeamBeamBiGaussian2D.compute_spacial_moments`, beambeam2d.py:272–283](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/beambeam2d.py#L272-L283)<br>[collective pipeline](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/beambeam2d.py#L238-L266) |

In 2D strong-strong beam-beam, `BeamBeamBiGaussian2D.compute_spacial_moments` sums every stored particle row without checking `particles.state`, so lost particles enter the population, centroid and size passed to the opposing beam. Removing a lost particle cannot change the collision, and with lost rows excluded the kicks agree with the analytic round-Gaussian field and the manual's beam-beam Lorentz factor. For four live 1 GeV/c protons at x, y = ±1 mm (weight 10¹⁰ each) plus one lost particle at x = 20 mm, the exchanged population is 5 × 10¹⁰ instead of 4 × 10¹⁰, the horizontal variance 6.480 × 10⁻⁵ m² instead of 1.000 × 10⁻⁶ m², and one opposing particle's horizontal kick +6.845 × 10⁻⁶ instead of −1.994 × 10⁻⁵. Strong-strong 2D simulations with particle losses are affected.

??? example "Reproducer: COL-05.py"

    [Download COL-05.py](COL-05.py)

    ```python
    """COL-05: Strong-strong 2D beam-beam moments include lost particles

    In 2D strong-strong beam-beam, `BeamBeamBiGaussian2D.compute_spacial_moments` sums every stored
    particle row without checking `particles.state`, so lost particles enter the population, centroid
    and size passed to the opposing beam. Removing a lost particle cannot change the collision, and with
    lost rows excluded the kicks agree with the analytic round-Gaussian field and the manual's beam-beam
    Lorentz factor. For four live 1 GeV/c protons at x, y = ±1 mm (weight 10¹⁰ each) plus one lost
    particle at x = 20 mm, the exchanged population is 5 × 10¹⁰ instead of 4 × 10¹⁰, the horizontal
    variance 6.480 × 10⁻⁵ m² instead of 1.000 × 10⁻⁶ m², and one opposing particle's horizontal kick
    +6.845 × 10⁻⁶ instead of −1.994 × 10⁻⁵. Strong-strong 2D simulations with particle losses are
    affected.

    The script collides two bunches for one turn through a two-branch pipeline. The source bunch holds
    the four live corner particles; one run adds the lost particle (state 0) and the control omits it,
    with the same four live target particles in both. The control's target kicks match the analytic
    round-Gaussian kick, and the run with the lost row is compared with them and with the exact source
    moments.
    """

    import numpy as np
    import xfields as xf
    import xtrack as xt

    RTOL, ATOL = 3e-13, 2e-16
    QELEM, EPSILON_0 = 1.60217662e-19, 8.854187817620e-12


    def evaluate():
        cases = {}
        for keep_lost in (False, True):
            manager = xt.PipelineManager()
            for name in ("a", "b"):
                manager.add_particles(name, 0)
            manager.add_element("ip")
            target = xt.Particles(
                p0c=1e9,
                x=[0.0008, -0.0004, -0.0004, 0.0008],
                y=[0.0008, 0.0008, -0.0006, -0.0006],
                weight=1e10,
            )
            source = xt.Particles(
                p0c=1e9,
                weight=1e10,
                x=[0.001, -0.001, -0.001, 0.001] + ([0.02] if keep_lost else []),
                y=[0.001, 0.001, -0.001, -0.001] + ([0.0] if keep_lost else []),
                state=[1, 1, 1, 1] + ([0] if keep_lost else []),
            )
            beta0, gamma0 = float(target.beta0[0]), float(target.gamma0[0])
            # Exact round-Gaussian field within the documented 2D soft-Gaussian model.
            factor = QELEM * 4e10 / target.mass0 * (1 + beta0**2) / (2 * gamma0 * beta0**2)
            radius2 = target.x**2 + target.y**2
            field_factor = -np.expm1(-radius2 / (2e-6)) / (2 * np.pi * EPSILON_0 * radius2)
            expected_px = factor * field_factor * target.x
            expected_py = factor * field_factor * target.y
            branches, lenses = [], []
            for own, other, particles in (("a", "b", target), ("b", "a", source)):
                particles.init_pipeline(own)
                lens = xf.BeamBeamBiGaussian2D(
                    other_beam_q0=1,
                    other_beam_beta0=beta0,
                    config_for_update=xf.ConfigForUpdateBeamBeamBiGaussian2D(
                        pipeline_manager=manager,
                        element_name="ip",
                        partner_particles_name=other,
                        update_every=1,
                    ),
                )
                line = xt.Line(elements=[lens])
                line.build_tracker()
                branches.append(xt.PipelineBranch(line, particles))
                lenses.append(lens)
            xt.PipelineMultiTracker(branches=branches).track(num_turns=1)
            assert np.all(target.state == 1)
            assert np.count_nonzero(source.state > 0) == 4
            observed = np.concatenate((target.px, target.py))
            expected = np.concatenate((expected_px, expected_py))
            if not keep_lost:
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
            else:
                cases["retained lost row, target px and py"] = observed, expected
                cases["retained lost row, exchanged source population"] = (
                    lenses[0].other_beam_num_particles,
                    4e10,
                )
                cases["retained lost row, exchanged source horizontal variance"] = (
                    lenses[0].other_beam_Sigma_11,
                    1e-6,
                )
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: removing a state=0 source row leaves beam-beam moments and kicks unchanged.")
        print(
            "The clean two-branch pipeline collision is checked against the analytic round-Gaussian Lorentz kick."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

??? example "Helper: _common.py"

    [Download _common.py](_common.py)

    ```python
    """Shared setup and output for the defect reproducers.

    `prepare()` moves to a fresh temporary working directory, where CFFI compiles the kernels, and
    prints the package versions and context; `report()` prints the observed and expected rows and the
    verdict line.
    """

    import atexit
    import importlib
    import importlib.metadata
    import os
    import platform
    import shutil
    import tempfile
    from pathlib import Path

    import numpy as np
    import xobjects as xo


    def prepare(*, radiation="off", random_seed=None):
        # CFFI compiles into cwd; keep generated files outside the source trees.
        scratch = Path(tempfile.mkdtemp(prefix="xsuite-repro-"))
        atexit.register(shutil.rmtree, scratch, ignore_errors=True)
        os.chdir(scratch)
        xo.settings.allow_kernel_compilation = True
        print(f"Python {platform.python_version()}; {platform.platform()}")
        for name in ("xsuite", "xtrack", "xobjects", "xdeps", "xpart", "xfields", "xcoll"):
            module = importlib.import_module(name)
            print(f"{name} {module.__version__}: {module.__file__}")
        for name in ("numpy", "scipy", "madng-tpsa", "cffi"):
            print(f"{name} {importlib.metadata.version(name)}")
        randomness = (
            "deterministic, no random seed"
            if random_seed is None
            else f"fixed RNG seed(s)={random_seed}"
        )
        print(f"ContextCpu, serial; radiation={radiation}; {randomness}")
        print(
            f"allow_kernel_compilation={xo.settings.allow_kernel_compilation}; "
            f"force_kernel_compilation={xo.settings.force_kernel_compilation}"
        )


    def report(cases, *, rtol, atol):
        np.set_printoptions(precision=15, linewidth=140)
        mismatch = False
        for label, (observed, expected) in cases.items():
            print(f"{label}: observed={np.asarray(observed)} expected={np.asarray(expected)}")
            mismatch |= not np.allclose(observed, expected, rtol=rtol, atol=atol)
        print(f"comparison: rtol={rtol:g}, atol={atol:g}")
        print("DEFECT REPRODUCED" if mismatch else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xfields 0.27.5 (`223b4fa134a1`), installed wheel and main worktree, CPU, serial |
| Confirmed | `gpt-6.1-sol` | low | 2026-10-26 | xfields 0.27.5 (`223b4fa134a1`), installed wheel and main worktree, CPU, serial |

