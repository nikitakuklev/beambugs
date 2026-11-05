---
title: "OPT-07: Periodic subrange slip factor uses the full parent line length"
hide:
  - toc
---

# OPT-07 — Periodic subrange slip factor uses the full parent line length

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`_add_periodic_solution_data_to_twiss_result`, twiss_postprocess_and_complem_results.py:30–71](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_postprocess_and_complem_results.py#L30-L71) |

Periodic Twiss on a subrange of a line divides the longitudinal response by the length of the whole parent line, corrupting the reported slip factor η and momentum compaction. Both are defined with the length of the selected periodic map, and the computed transfer matrix agrees with that local map. For a 2 m `LineSegmentMap` with momentum compaction 0.02, zero RF voltage and a 1 GeV/c proton reference, adding an untracked 10 m prefix changes η from −0.4482 to −0.07470 and the momentum compaction from 0.02000 to 0.3935, although these dimensionless quantities should not change. This occurs for constant-energy 4D periodic subranges with unused prefixes or suffixes; the selected matrix itself stays correct.

??? example "Reproducer: OPT-07.py"

    [Download OPT-07.py](OPT-07.py)

    ```python
    """OPT-07: Periodic subrange slip factor uses the full parent line length

    Periodic Twiss on a subrange of a line divides the longitudinal response by the length of the whole
    parent line, corrupting the reported slip factor η and momentum compaction. Both are defined with
    the length of the selected periodic map, and the computed transfer matrix agrees with that local
    map. For a 2 m `LineSegmentMap` with momentum compaction 0.02, zero RF voltage and a 1 GeV/c proton
    reference, adding an untracked 10 m prefix changes η from −0.4482 to −0.07470 and the momentum
    compaction from 0.02000 to 0.3935, although these dimensionless quantities should not change. This
    occurs for constant-energy 4D periodic subranges with unused prefixes or suffixes; the selected
    matrix itself stays correct.

    The script places that `LineSegmentMap` after a drift of length 0 or 10 m, ending at a marker, and
    runs a periodic `twiss4d` over the segment alone. It first checks that the matrix element R₅₆
    (`zeta` response to `pzeta`) equals −2η, with η = α − 1/γ₀² for momentum compaction α and reference
    Lorentz factor γ₀, then compares the reported `slip_factor` and `momentum_compaction_factor` with η
    and 0.02; the drift-free case is the control.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 0.0, 5e-10


    def evaluate():
        cases = {}
        for prefix in (0, 10):
            line = xt.Line(
                elements={
                    "lead": xt.Drift(length=prefix),
                    "arc": xt.LineSegmentMap(
                        length=2,
                        qx=0.2,
                        qy=0.3,
                        longitudinal_mode="nonlinear",
                        voltage_rf=0,
                        frequency_rf=1e8,
                        momentum_compaction_factor=0.02,
                    ),
                    "end": xt.Marker(),
                }
            )
            line.particle_ref = xt.Particles(p0c=1e9)
            expected_eta = 0.02 - 1 / line.particle_ref.gamma0[0] ** 2
            tw = line.twiss4d(start="arc", end="end", init="periodic", chrom=False)
            # At delta=0, dpzeta/ddelta=1. The tracked map itself has correct slippage.
            np.testing.assert_allclose(tw.R_matrix[4, 5], -2 * expected_eta, rtol=0, atol=2 * ATOL)
            # Normalizing by the 12 m parent line instead of the 2 m map scales eta by 2/12;
            # alpha_c = eta + 1/gamma0**2 then inherits the error.
            observed = np.array([tw.slip_factor, tw.momentum_compaction_factor])
            expected = np.array([expected_eta, 0.02])
            if prefix == 0:
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
            cases[f"untracked prefix length={prefix}, slip_factor,alpha_c"] = observed, expected
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: eta=-R[zeta,pzeta]/2 and alpha_c=eta+1/gamma0²=0.02 for the selected 2 m periodic map."
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
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

