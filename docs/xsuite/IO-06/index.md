---
title: "IO-06: Negative-charge reference rigidity fails a constructor round trip"
hide:
  - toc
---

# IO-06 — Negative-charge reference rigidity fails a constructor round trip

| | |
|---|---|
| Code | [Xsuite](../index.md) · Lattices, particles and I/O |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`Particles._update_refs`, particles.py:1704–1708](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L1704-L1708)<br>[`rigidity0` getter and linked setter](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/particles/particles.py#L1342-L1360) |

Constructing `Particles` from the reference rigidity `rigidity0` computes the momentum from `rigidity0`·|q₀|·c, while the `rigidity0` getter and setter use the signed reference charge q₀, so a negative-charge reference's own reported rigidity reconstructs a negative momentum. The manual defines signed rigidity, and property-to-constructor round trips must preserve the reference state. For electron-mass references with q₀ = −1 or −2 at 1 GeV/c, the constructor returns the reference momentum `p0c` = −1.000×10⁹ eV and reference speed β₀ = v/c = −1.000, while the particle remains alive; the correct momentum and speed are positive. Positive-charge construction, the linked rigidity setter, and construction from `p0c` or `energy0` pass their controls.

??? example "Reproducer: IO-06.py"

    [Download IO-06.py](IO-06.py)

    ```python
    """IO-06: Negative-charge reference rigidity fails a constructor round trip

    Constructing `Particles` from the reference rigidity `rigidity0` computes the momentum from
    `rigidity0`·|q₀|·c, while the `rigidity0` getter and setter use the signed reference charge q₀, so a
    negative-charge reference's own reported rigidity reconstructs a negative momentum. The manual
    defines signed rigidity, and property-to-constructor round trips must preserve the reference state.
    For electron-mass references with q₀ = −1 or −2 at 1 GeV/c, the constructor returns the reference
    momentum `p0c` = −1.000×10⁹ eV and reference speed β₀ = v/c = −1.000, while the particle remains
    alive; the correct momentum and speed are positive. Positive-charge construction, the linked
    rigidity setter, and construction from `p0c` or `energy0` pass their controls.

    For electron-mass references with q₀ = ±1 and ±2 at 1 GeV/c, the script reads `rigidity0`, checks it
    against the signed value p₀c/(c·q₀), assigns it back through the linked setter and passes it to a
    new `Particles` with unchanged mass and charge. The rebuilt `p0c` and `beta0` must equal the
    originals. The setter and the positive charges are controls; the negative charges show the bug.
    """

    import numpy as np
    import xtrack as xt
    from scipy.constants import c

    RTOL, ATOL = 2e-15, 0


    def evaluate():
        cases = {}
        for charge in (1, -1, 2, -2):
            original = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=charge, p0c=1e9)
            rigidity = float(original.rigidity0[0])
            # The getter and linked setter use signed rigidity, p0c/(c*q0).
            np.testing.assert_allclose(rigidity, 1e9 / (c * charge), rtol=RTOL, atol=ATOL)
            assigned = original.copy()
            assigned.rigidity0 = rigidity
            np.testing.assert_allclose(assigned.p0c, original.p0c, rtol=RTOL, atol=ATOL)
            rebuilt = xt.Particles(mass0=original.mass0, q0=charge, rigidity0=rigidity)
            assert rebuilt.state[0] == 1
            if charge > 0:
                np.testing.assert_allclose(rebuilt.p0c, original.p0c, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(rebuilt.beta0, original.beta0, rtol=RTOL, atol=ATOL)
            else:
                cases[f"q0={charge}, reconstructed p0c [eV]"] = rebuilt.p0c.copy(), original.p0c.copy()
                cases[f"q0={charge}, reconstructed beta0"] = rebuilt.beta0.copy(), original.beta0.copy()
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected: constructing from the same object's rigidity0 preserves p0c and beta0, as assigning that property does."
        )
        print(
            "Signed Lorentz rigidity and positive-charge controls are checked first; no convention change can justify a failed round trip."
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
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

