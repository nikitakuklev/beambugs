---
title: "COL-07: Loss-map periodic aperture cells have negative lengths"
hide:
  - toc
---

# COL-07 — Loss-map periodic aperture cells have negative lengths

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xcoll main (`64e13f5`) |
| Source | [`LossMap._create_aperture_pos_dict`, lossmap.py:863–888](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/lossmap.py#L863-L888)<br>[loss-density normalization](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/plot.py#L215-L230) |

For a non-interpolated loss map (`interpolation=False`), `LossMap` bounds each aperture cell by the midpoints to its neighbours without wrapping the first and last neighbours around the ring. Those two cells get negative lengths and displaced centres, whereas periodic midpoint cells must have positive lengths summing to the circumference. For apertures at 1, 4 and 9 m in a 12 m ring, the lengths are −2.5, 4 and −1.5 m instead of 3.5, 4 and 4.5 m, summing to 0 instead of 12 m, and the end-cell centres are 3.75 and 5.75 m instead of 0.75 and 8.75 m. Non-interpolated ring loss maps report wrong loss densities and positions in the boundary cells; interior cells are correct.

??? example "Reproducer: COL-07.py"

    [Download COL-07.py](COL-07.py)

    ```python
    """COL-07: Loss-map periodic aperture cells have negative lengths

    For a non-interpolated loss map (`interpolation=False`), `LossMap` bounds each aperture cell by the
    midpoints to its neighbours without wrapping the first and last neighbours around the ring. Those
    two cells get negative lengths and displaced centres, whereas periodic midpoint cells must have
    positive lengths summing to the circumference. For apertures at 1, 4 and 9 m in a 12 m ring, the
    lengths are −2.5, 4 and −1.5 m instead of 3.5, 4 and 4.5 m, summing to 0 instead of 12 m, and the
    end-cell centres are 3.75 and 5.75 m instead of 0.75 and 8.75 m. Non-interpolated ring loss maps
    report wrong loss densities and positions in the boundary cells; interior cells are correct.

    The script tracks three 2 GeV/c protons into the three apertures, checks where they were lost, and
    builds the loss map with `correct_aperture_absorption=False` and no material or scattering engine.
    It compares the cell lengths and centres with periodic midpoint cells built from the positive cyclic
    gaps between apertures. It requires the installed prebuilt kernels (compilation is forbidden).
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
    # A periodic Voronoi cell has a left half-gap and a right half-gap. This
    # construction uses positive cyclic differences, without the source's neighbours.
    L = 12.0
    aperture_s = np.array([1.0, 4.0, 9.0])
    forward_gaps = np.mod(np.roll(aperture_s, -1) - aperture_s, L)
    backward_gaps = np.roll(forward_gaps, 1)
    expected_lengths = (backward_gaps + forward_gaps) / 2
    expected_centres = np.mod(aperture_s + (forward_gaps - backward_gaps) / 4, L)
    data = forward.aperture_losses
    assert list(data["name"]) == ["ap0", "ap1", "ap2"]
    print(
        "LENGTHS observed=",
        data["length"],
        "expected=",
        expected_lengths,
        "sum=",
        data["length"].sum(),
        "circumference=",
        L,
    )
    print("CENTRES observed=", data["s"], "expected=", expected_centres)
    np.testing.assert_array_equal(expected_lengths, [3.5, 4.0, 4.5])
    assert expected_lengths.sum() == L

    kernels()
    report(
        {
            "aperture lengths": (data["length"], expected_lengths),
            "aperture centres": (data["s"], expected_centres),
        },
        rtol=0,
        atol=1e-13,
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

