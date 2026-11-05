---
title: "COL-09: Noninterpolated aperture losses are omitted from total energy"
hide:
  - toc
---

# COL-09 — Noninterpolated aperture losses are omitted from total energy

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xcoll main (`64e13f5`) |
| Source | [`LossMap.tot_energy_aperture`, lossmap.py:140–154](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/lossmap.py#L140-L154) |

In a non-interpolated loss map (`interpolation=False`), `LossMap.tot_energy_aperture` sums the binned aperture store, although this mode records aperture energies in unbinned storage, so it returns zero and `tot_energy` omits all aperture losses. With no collimator losses both totals must equal the sum of the per-aperture energies that `aperture_losses` reports, which match the mass-shell energy 3·√((p₀c)² + m²), with p₀ the momentum and m the proton rest energy. For three 2 GeV/c protons lost at the apertures of a 12 m ring, the per-aperture energies sum to 6.627 × 10⁹ eV, while `tot_energy_aperture` and `tot_energy` are both 0.

??? example "Reproducer: COL-09.py"

    [Download COL-09.py](COL-09.py)

    ```python
    """COL-09: Noninterpolated aperture losses are omitted from total energy

    In a non-interpolated loss map (`interpolation=False`), `LossMap.tot_energy_aperture` sums the
    binned aperture store, although this mode records aperture energies in unbinned storage, so it
    returns zero and `tot_energy` omits all aperture losses. With no collimator losses both totals must
    equal the sum of the per-aperture energies that `aperture_losses` reports, which match the
    mass-shell energy 3·√((p₀c)² + m²), with p₀ the momentum and m the proton rest energy. For three 2
    GeV/c protons lost at the apertures of a 12 m ring, the per-aperture energies sum to 6.627 × 10⁹ eV,
    while `tot_energy_aperture` and `tot_energy` are both 0.

    The script tracks three 2 GeV/c protons into apertures at 1, 4 and 9 m and builds the loss map with
    `correct_aperture_absorption=False`. It checks the per-aperture energies against the mass-shell sum
    and that no collimator energy is recorded, then compares both public totals with that sum. It
    requires the installed prebuilt kernels (compilation is forbidden).
    """

    from _prebuilt import CTX, identity, kernels, np, report, xc, xt

    identity()
    line = xt.Line(
        elements=[
            xt.Drift(length=1),
            xt.LimitRect(min_x=-0.004, max_x=0.004),
            xt.Drift(length=3),
            xt.LimitRect(min_x=-0.002, max_x=0.002),
            xt.Drift(length=5),
            xt.LimitRect(min_x=-0.0007, max_x=0.0007),
            xt.Drift(length=3),
        ],
        element_names=["d0", "ap0", "d1", "ap1", "d2", "ap2", "d3"],
    )
    line.particle_ref = xt.Particles(p0c=2e9, pdg_id=2212)
    line.build_tracker(_context=CTX)
    p = xt.Particles(_context=CTX, p0c=2e9, pdg_id=2212, x=[0.005, 0.003, 0.001])
    line.track(p)
    order = np.argsort(p.particle_id)
    print(
        "TRACKED states=",
        np.asarray(p.state)[order],
        "at_element=",
        np.asarray(p.at_element)[order],
        "s=",
        np.asarray(p.s)[order],
    )
    np.testing.assert_array_equal(np.asarray(p.state)[order], [0, 0, 0])
    np.testing.assert_array_equal(np.asarray(p.at_element)[order], [1, 3, 5])
    np.testing.assert_array_equal(np.asarray(p.s)[order], [1, 4, 9])


    def lossmap(part, reversed=False):
        return xc.LossMap(
            line=line,
            part=part,
            line_is_reversed=reversed,
            interpolation=False,
            correct_aperture_absorption=False,
            verbose=False,
        )


    forward = lossmap(p.copy())
    data = forward.aperture_losses
    # Part.energy and the per-aperture arrays agree with the mass shell, but the
    # total should also report their sum. No material/scattering engine is involved.
    expected_energy = 3 * np.hypot(2e9, p.mass0)
    print(
        f"PER_APERTURE sum={data['e'].sum():.17g} "
        f"mass_shell_expected={expected_energy:.17g} "
        f"tot_energy_aperture={forward.tot_energy_aperture:.17g} "
        f"tot_energy={forward.tot_energy:.17g}"
    )
    np.testing.assert_allclose(data["e"].sum(), expected_energy, rtol=3e-15, atol=0)
    assert forward.num_aperture_losses == 3
    assert forward.tot_energy_collimator == 0


    kernels()
    report(
        {
            "total aperture energy": (forward.tot_energy_aperture, expected_energy),
            "total loss energy": (forward.tot_energy, expected_energy),
        },
        rtol=3e-15,
        atol=0,
    )

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

