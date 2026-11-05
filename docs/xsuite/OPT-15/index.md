---
title: "OPT-15: Closest tune approach is inflated across an integer boundary"
hide:
  - toc
---

# OPT-15 — Closest tune approach is inflated across an integer boundary

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack main (`f09e0fb`) |
| Source | [`_add_periodic_solution_data_to_twiss_result`, twiss_postprocess_and_complem_results.py:99–148](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_postprocess_and_complem_results.py#L99-L148)<br>[closest tune approach coefficient](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L333-L337)<br>[coupling manual](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/linear_coupling.tex#L139-L174) |

Twiss inflates `c_minus`, the closest-tune-approach coefficient, when two nearby eigentunes straddle an integer. An analytic coupled-oscillator calculation and an independent eigenfrequency scan show that shifting both tunes together must preserve their minimum separation. For eigentunes 0.9920 and 1.006, mixing angle 0.07 rad and modal beta function 2.7 m, `c_minus` is 0.1376 instead of 0.001954, too large by a factor 70.43; shifting both tunes by ±0.09 restores the correct value. The error appears in the 4D and 6D periodic calculations with positive line length and pure difference coupling, including the complex coupling magnitudes; tracking maps, eigentunes and projected beta functions remain correct.

??? example "Reproducer: OPT-15.py"

    [Download OPT-15.py](OPT-15.py)

    ```python
    """OPT-15: Closest tune approach is inflated across an integer boundary

    Twiss inflates `c_minus`, the closest-tune-approach coefficient, when two nearby eigentunes straddle
    an integer. An analytic coupled-oscillator calculation and an independent eigenfrequency scan show
    that shifting both tunes together must preserve their minimum separation. For eigentunes 0.9920 and
    1.006, mixing angle 0.07 rad and modal beta function 2.7 m, `c_minus` is 0.1376 instead of 0.001954,
    too large by a factor 70.43; shifting both tunes by ±0.09 restores the correct value. The error
    appears in the 4D and 6D periodic calculations with positive line length and pure difference
    coupling, including the complex coupling magnitudes; tracking maps, eigentunes and projected beta
    functions remain correct.

    The script builds rings of 24 or 48 rotated `LineSegmentMap` elements (modal beta 2.7 m, 0.8 GeV/c
    protons) with pure difference coupling at mixing angles θ = 0.07, 0.31 and 0.53 rad, shifts both
    tunes by −0.09, 0 and +0.09, and runs 4D and 6D Twiss. The exact closest tune approach, the
    eigentune gap times |sin 2θ|, is confirmed by a 141-point eigenvalue scan and compared with
    `c_minus`; tracked maps, tunes and projected beta functions are checked as well. Cases that do not
    straddle an integer are the controls.
    """

    import json

    from _prebuilt import CTX, identity, kernels, np, xt

    identity()


    def rotation(tune, beta):
        c, s = np.cos(2 * np.pi * tune), np.sin(2 * np.pi * tune)
        return np.array([[c, beta * s], [-s / beta, c]])


    results = []
    for center, gap, theta in ((0.999, 0.014, 0.07), (1.004, 0.038, 0.31), (2.002, 0.022, 0.53)):
        mix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        h = mix.T @ np.diag([center - gap / 2, center + gap / 2]) @ mix
        k = h[0, 1]
        detunings = np.linspace(-0.07, 0.07, 141)
        gaps = [
            np.diff(np.linalg.eigvalsh([[center + d / 2, k], [k, center - d / 2]]))[0]
            for d in detunings
        ]
        # For H=[[a,k],[k,b]], the eigentune gap is sqrt((a-b)^2+4*k^2).
        # Its minimum at fixed k is 2*abs(k), invariant under common tune shifts.
        expected = float(2 * abs(k))
        assert abs(min(gaps) - expected) < 2e-15
        for shift in (-0.09, 0.0, 0.09):
            q1, q2 = center - gap / 2 + shift, center + gap / 2 + shift
            for nseg in (24, 48):
                line = xt.Line(
                    elements=[
                        xt.LineSegmentMap(
                            length=2.0,
                            qx=q1 / nseg,
                            qy=q2 / nseg,
                            betx=2.7,
                            bety=2.7,
                            qs=0.073 / nseg,
                            bets=11.0,
                            rot_s_rad=theta,
                        )
                        for _ in range(nseg)
                    ]
                )
                line.particle_ref = xt.Particles(_context=CTX, p0c=0.8e9)
                line.build_tracker(_context=CTX)
                # Track all transverse Cartesian basis directions with both signs.
                rays = np.vstack([np.eye(4), -np.eye(4)]) * 1e-5
                p = xt.Particles(
                    _context=CTX, p0c=0.8e9, x=rays[:, 0], px=rays[:, 1], y=rays[:, 2], py=rays[:, 3]
                )
                line.track(p)
                assert np.all(p.state == 1)
                diag = np.zeros((4, 4))
                diag[:2, :2] = rotation(q1, 2.7)
                diag[2:, 2:] = rotation(q2, 2.7)
                m = np.kron(mix.T, np.eye(2)) @ diag @ np.kron(mix, np.eye(2))
                actual_rays = np.array([p.x, p.px, p.y, p.py]).T
                track_error = float(np.max(np.abs(actual_rays - rays @ m.T)))
                assert track_error < 1e-17
                for method in ("4d", "6d"):
                    for el in line.elements:
                        el.qs = 0.0 if method == "4d" else 0.073 / nseg
                    tw = line.twiss(method=method, chrom=False)
                    assert max(abs(tw.qx - q1), abs(tw.qy - q2)) < 1e-12
                    beta_error = max(
                        abs(tw.betx[0] - 2.7 * np.cos(theta) ** 2),
                        abs(tw.betx2[0] - 2.7 * np.sin(theta) ** 2),
                    )
                    assert beta_error < 1e-11
                    straddles = int(np.floor(q1)) != int(np.floor(q2))
                    ratio = tw.c_minus / expected
                    assert (
                        np.max(np.abs(np.abs(tw.c_minus_re + 1j * tw.c_minus_im) - tw.c_minus)) < 1e-12
                    )
                    assert np.max(np.abs(np.abs(tw.c_minus_local) - tw.c_minus)) < 1e-12
                    if straddles:
                        # c_minus uses |frac(qx) - frac(qy)|, which turns a gap across an integer
                        # into 1 - gap and so inflates c_minus by (1 - gap)/gap.
                        assert abs(ratio - (1 - gap) / gap) < 2e-9
                    else:
                        assert abs(tw.c_minus - expected) < 1e-12
                    rec = dict(
                        center=center,
                        gap=gap,
                        theta=theta,
                        shift=shift,
                        nseg=nseg,
                        method=method,
                        qx=tw.qx,
                        qy=tw.qy,
                        expected=expected,
                        actual=tw.c_minus,
                        ratio=ratio,
                        eigenscan_min=float(min(gaps)),
                        track_error=track_error,
                        beta_error=float(beta_error),
                        straddles=straddles,
                    )
                    results.append(rec)
                    print("COUPLING", json.dumps(rec))
                    if nseg == 24 and method == "6d":
                        print(
                            f"TUNES q1={q1:g} q2={q2:g} expected={expected:.17g} "
                            f"actual={tw.c_minus:.17g} ratio={ratio:.12g}"
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

