---
title: "OPT-13: Reversed Twiss tables reconstruct incorrect transverse transfer maps"
hide:
  - toc
---

# OPT-13 — Reversed Twiss tables reconstruct incorrect transverse transfer maps

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [`TwissTable.reverse`, twiss_table.py:984–1111](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_table.py#L984-L1111)<br>[matrix reconstruction APIs, lines 775–881](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_table.py#L775-L881) |

Transverse transfer matrices reconstructed from `TwissTable.reverse()` or `Line.twiss(reverse=True)` give the wrong reversed transport, including spurious focusing in a drift. Analytic drift and thin-lens matrices, transformed into the documented reverse frame, agree with native backtracking and require a reversed drift to keep a positive length. For a 1.7 GeV/c proton line containing 0.7600 m of drift, with initial horizontal beta function 0.7 m and alpha −0.8, the reconstructed response of position to normalized momentum, R₁₂, is −0.7600 m instead of +0.7600 m, and the response of momentum to position, R₂₁, is 2.035 m⁻¹ instead of zero. Both reconstruction APIs and marker subintervals are affected in the tested uncoupled channels; forward reconstruction passes.

??? example "Reproducer: OPT-13.py"

    [Download OPT-13.py](OPT-13.py)

    ```python
    """OPT-13: Reversed Twiss tables reconstruct incorrect transverse transfer maps

    Transverse transfer matrices reconstructed from `TwissTable.reverse()` or `Line.twiss(reverse=True)`
    give the wrong reversed transport, including spurious focusing in a drift. Analytic drift and
    thin-lens matrices, transformed into the documented reverse frame, agree with native backtracking
    and require a reversed drift to keep a positive length. For a 1.7 GeV/c proton line containing
    0.7600 m of drift, with initial horizontal beta function 0.7 m and alpha −0.8, the reconstructed
    response of position to normalized momentum, R₁₂, is −0.7600 m instead of +0.7600 m, and the
    response of momentum to position, R₂₁, is 2.035 m⁻¹ instead of zero. Both reconstruction APIs and
    marker subintervals are affected in the tested uncoupled channels; forward reconstruction passes.

    The script builds four open 1.7 GeV/c proton lines with drifts L, 0.7L and 0.2L, thin lenses of
    strength k and −0.4k (k = 0, +0.27 and −0.19 m⁻¹) and three markers. In the reverse frame, with F =
    diag(−1, 1, 1, −1), a forward map M becomes F·M⁻¹·F; native backtracking of signed basis rays
    confirms this analytic map, which is compared with both table matrix methods over each marker
    interval and with `Line.twiss(reverse=True)`. Forward reconstruction and double reversal are the
    controls, and negating the stored reverse phases in a copy of the table restores agreement.
    """

    import json

    from _prebuilt import CTX, identity, kernels, np, xt

    identity()
    # The documented reverse frame sends proper transport M to F @ inv(M) @ F.
    F = np.diag([-1.0, 1.0, 1.0, -1.0])


    def drift(length):
        m = np.eye(4)
        m[0, 1] = m[2, 3] = length
        return m


    def lens(strength):
        m = np.eye(4)
        m[1, 0], m[3, 2] = -strength, strength
        return m


    results = []
    for L, strength, bx, ax, by, ay in (
        (0.4, 0.0, 0.7, -0.8, 6.0, 0.6),
        (3.1, 0.0, 8.0, 1.3, 1.2, -1.1),
        (1.3, 0.27, 2.3, -0.6, 4.7, 0.9),
        (2.1, -0.19, 3.4, 0.5, 0.8, -0.4),
    ):
        elements = [
            xt.Marker(),
            xt.Drift(length=L),
            xt.Multipole(knl=[0.0, strength]),
            xt.Marker(),
            xt.Drift(length=0.7 * L),
            xt.Multipole(knl=[0.0, -0.4 * strength]),
            xt.Drift(length=0.2 * L),
            xt.Marker(),
        ]
        line = xt.Line(elements=elements, element_names=["a", "d1", "q1", "b", "d2", "q2", "d3", "c"])
        line.particle_ref = xt.Particles(_context=CTX, p0c=1.7e9)
        line.build_tracker(_context=CTX)
        prefix = [np.eye(4)]
        for m in [
            np.eye(4),
            drift(L),
            lens(strength),
            np.eye(4),
            drift(0.7 * L),
            lens(-0.4 * strength),
            drift(0.2 * L),
            np.eye(4),
        ]:
            prefix.append(m @ prefix[-1])
        full = prefix[-1]
        expected = F @ np.linalg.inv(full) @ F
        # Symmetric Cartesian basis rays measure the linearized map independently.
        eps = 1e-7
        rays = np.vstack([eps * np.eye(4), -eps * np.eye(4)])
        proper_in = rays @ F
        p = xt.Particles(
            _context=CTX,
            p0c=1.7e9,
            x=proper_in[:, 0],
            px=proper_in[:, 1],
            y=proper_in[:, 2],
            py=proper_in[:, 3],
        )
        line.track(p, backtrack=True)
        assert np.all(p.state == 1)
        got = np.array([p.x, p.px, p.y, p.py]).T @ F
        native_error = float(np.max(np.abs(got - rays @ expected.T)))
        assert native_error < 2e-19
        tw = line.twiss(method="4d", betx=bx, alfx=ax, bety=by, alfy=ay, chrom=False)
        rev = tw.reverse()
        actual = rev.get_R_matrix("c", "a")[:4, :4]
        forward_error = float(np.max(np.abs(tw.get_R_matrix("a", "c")[:4, :4] - full)))
        assert forward_error < 3e-13
        table_error = float(np.max(np.abs(rev.get_R_matrix_table().R_matrix[-2, :4, :4] - actual)))
        # Last marker is 'a'; endpoint has the same phase-space boundary.
        assert table_error < 3e-13
        oracle_error = float(np.max(np.abs(actual - expected)))
        assert oracle_error > 0.1
        corrected = rev._copy()
        corrected._data["phix"] = -rev.phix
        corrected._data["phiy"] = -rev.phiy
        repair_error = float(np.max(np.abs(corrected.get_R_matrix("c", "a")[:4, :4] - expected)))
        assert repair_error < 3e-13
        double_error = float(np.max(np.abs(rev.reverse().get_R_matrix("a", "c")[:4, :4] - full)))
        assert double_error < 3e-13
        public = line.twiss(method="4d", reverse=True, betx=bx, alfx=ax, bety=by, alfy=ay, chrom=False)
        public_error = float(np.max(np.abs(public.get_R_matrix("c", "a")[:4, :4] - expected)))
        assert public_error > 0.1
        # Every marker interval checks row association and local versus total phase.
        intervals = {}
        for start, end in [("c", "b"), ("b", "a"), ("c", "a")]:
            ia = line.element_names.index(end)
            ib = line.element_names.index(start)
            local = F @ np.linalg.inv(prefix[ib] @ np.linalg.inv(prefix[ia])) @ F
            intervals[start + ":" + end] = float(
                np.max(np.abs(rev.get_R_matrix(start, end)[:4, :4] - local))
            )
        rec = dict(
            L=L,
            strength=strength,
            expected=expected.tolist(),
            actual=actual.tolist(),
            forward_error=forward_error,
            native_error=native_error,
            repair_error=repair_error,
            double_error=double_error,
            oracle_error=oracle_error,
            public_reverse_error=public_error,
            interval_errors=intervals,
        )
        assert min(intervals.values()) > 0.1
        results.append(rec)
        print("REVERSE", json.dumps(rec))
        print(
            f"MAP L={L:g} strength={strength:g} R12_expected={expected[0, 1]:.17g} "
            f"R12_actual={actual[0, 1]:.17g} R21_expected={expected[1, 0]:.17g} "
            f"R21_actual={actual[1, 0]:.17g}"
        )
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
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels (compilation forbidden), CPU, serial |

