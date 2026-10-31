---
title: "COL-13: Everest deposited-energy accumulators omit macroparticle weights"
hide:
  - toc
---

# COL-13 — Everest deposited-energy accumulators omit macroparticle weights

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xcoll main (`64e13f5`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`EverestCollimator_track_local_particle`, everest_collimator.h:226–234](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/beam_elements/elements_src/everest_collimator.h#L226-L234)<br>[Geant4 accumulator](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/scattering_routines/geant4/track.py#L128-L129) |

`EverestCollimator` adds each particle's energy loss to its deposited-energy counters without the macroparticle `weight`, the number of physical particles a macroparticle represents. The deposited energy is the sum of `weight` times energy loss, the weighting Xcoll's Geant4 interface applies to the same counters. For 20 protons at 400 GeV/c through 0.6 mm of carbon-fibre composite with fixed random seeds, weights 1 to 5 give a primary deposit of 2.865 × 10⁶ eV, the same as unit weights, instead of 8.653 × 10⁶ eV. Simulations with non-unit macroparticle weights are affected; by source reading, the secondary-particle counter, `EverestBlock` and `EverestCrystal` contain the same expression.

??? example "Reproducer: COL-13.py"

    [Download COL-13.py](COL-13.py)

    ```python
    """COL-13: Everest deposited-energy accumulators omit macroparticle weights

    `EverestCollimator` adds each particle's energy loss to its deposited-energy counters without the
    macroparticle `weight`, the number of physical particles a macroparticle represents. The deposited
    energy is the sum of `weight` times energy loss, the weighting Xcoll's Geant4 interface applies to
    the same counters. For 20 protons at 400 GeV/c through 0.6 mm of carbon-fibre composite with fixed
    random seeds, weights 1 to 5 give a primary deposit of 2.865 × 10⁶ eV, the same as unit weights,
    instead of 8.653 × 10⁶ eV. Simulations with non-unit macroparticle weights are affected; by source
    reading, the secondary-particle counter, `EverestBlock` and `EverestCrystal` contain the same
    expression.

    The script tracks 20 protons from x = 3 mm through the collimator (jaw 1 mm) with per-particle seeds
    7001–7020, all surviving, once with unit weights and once with weights 1, 2, 3, 4, 5 repeated four
    times. The per-particle energy losses are identical in both runs; the script checks that the
    accumulated deposit equals their unweighted sum and compares it with the weighted sum. It requires
    the installed prebuilt kernels (compilation is forbidden).
    """

    from _prebuilt import CTX, identity, kernels, np, xc, xt

    identity(random_seed="7001..7020")

    n = 20
    p0c = 4e11
    seeds = np.arange(7001, 7001 + n, dtype=np.uint32)


    def run(label, weights, state):
        coll = xc.EverestCollimator(
            _context=CTX,
            length=0.0006,
            jaw=0.001,
            material=xc.materials.CarbonFibreComposite,
            mark_scattered_particles=True,
        )
        line = xt.Line(elements={"carbon": coll, "drift": xt.Drift(length=0.2)})
        line.particle_ref = xt.Particles(_context=CTX, p0c=p0c, pdg_id_0=2212)
        line.build_tracker(_context=CTX)
        particles = xt.Particles(
            _context=CTX, p0c=p0c, pdg_id_0=2212, x=np.full(n, 0.003), weight=weights, state=state
        )
        particles._init_random_number_generator(seeds=seeds)
        initial = np.array(particles.energy)
        particles.reorganize()
        line.xcoll.scattering.enable()
        line.track(particles, num_turns=1)
        particles.sort(interleave_lost_particles=True)
        np.testing.assert_array_equal(particles.particle_id, np.arange(n))
        assert np.all(particles.state > 0), "Need a survival-only case to isolate deposition"
        energy_loss = initial - np.array(particles.energy)
        assert np.all(energy_loss > 0)
        # Xobjects generates the accumulator fields at class creation.
        primary = float(coll._acc_ionisation_loss)  # ty: ignore[unresolved-attribute]
        secondary = float(coll._acc_ionisation_loss_sec)  # ty: ignore[unresolved-attribute]
        recorded = primary + secondary
        physical = np.dot(energy_loss, np.array(particles.weight))
        # At these energies one energy subtraction has O(1e-4 eV) roundoff.
        # Use a conservative bound from the floating-point spacing, not a percent tolerance.
        atol = n * 4 * np.spacing(p0c)
        np.testing.assert_allclose(recorded, energy_loss.sum(), rtol=0, atol=atol)
        lossmap = xc.LossMap(
            line=line,
            part=particles,
            weights=np.array(particles.weight),
            interpolation=False,
            correct_aperture_absorption=False,
            verbose=False,
        )
        assert lossmap.summary.n.sum() == 0
        np.testing.assert_allclose(lossmap.tot_energy_collimator, recorded, rtol=0, atol=0)
        print(
            f"{label}: primary_counter={primary:.17g} secondary_counter={secondary:.17g} unweighted_dE={energy_loss.sum():.17g} weighted_dE={physical:.17g}"
        )
        return recorded, physical, energy_loss


    unit = run("unit-primary", np.ones(n), 1)
    weighted = run("weighted-primary", 1.0 + np.arange(n) % 5, 1)
    np.testing.assert_array_equal(unit[2], weighted[2])
    assert unit[0] == weighted[0]
    assert abs(weighted[0] - weighted[1]) > 1e6
    print("IDENTICAL energy losses and accumulator with unit and nonuniform weights")
    kernels()
    print("DEFECT REPRODUCED")

    ```

??? example "Helper: _prebuilt.py"

    [Download _prebuilt.py](_prebuilt.py)

    ```python
    """Serial CPU setup that runs Xsuite's prebuilt kernels with kernel compilation forbidden.

    `identity()` prints the package versions and context; `kernels()` asserts that a prebuilt kernel
    library was loaded and prints it.
    """

    import importlib
    import importlib.metadata
    import os
    import platform
    from pathlib import Path

    # Configure thread pools before importing NumPy or Xsuite.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"
    os.environ["XSUITE_ALLOW_KERNEL_COMPILATION"] = "0"
    os.environ["XSUITE_FORCE_KERNEL_COMPILATION"] = "0"
    os.environ["XSUITE_CFFI_FORBID_COMPILE"] = "1"

    import numpy as np
    import xcoll as xc
    import xobjects as xo
    import xtrack as xt
    from _common import report

    __all__ = ["CTX", "identity", "kernels", "np", "particles", "report", "xc", "xt"]

    np.set_printoptions(precision=17)
    xo.settings.allow_kernel_compilation = False
    xo.settings.force_kernel_compilation = False
    CTX = xo.ContextCpu(omp_num_threads=0)


    def identity(*, radiation="off", random_seed=None):
        print(f"Python {platform.python_version()}; {platform.platform()}")
        for name in ("numpy", "scipy", "cffi"):
            print(name, importlib.metadata.version(name))
        for name in ("xsuite", "xtrack", "xobjects", "xpart", "xfields", "xdeps", "xcoll"):
            mod = importlib.import_module(name)
            print("PACKAGE", name, mod.__version__, mod.__file__)
        randomness = (
            "deterministic, no random seed"
            if random_seed is None
            else f"fixed RNG seed(s)={random_seed}"
        )
        print(
            f"CONTEXT CPU serial; radiation {radiation}; {randomness}; "
            "kernel compilation forbidden; prebuilt kernels"
        )


    def kernels():
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "/xsuite/lib/" in line and ".so" in line
            }
        )
        assert libraries, "No prebuilt kernel library loaded"
        print("KERNEL_LIBRARIES", libraries)


    def particles(**kwargs):
        return xt.Particles(_context=CTX, p0c=1e9, **kwargs)

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
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

