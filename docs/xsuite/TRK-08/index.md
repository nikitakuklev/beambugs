---
title: "TRK-08: SplineBoris omits species factors in physical charge and momentum"
hide:
  - toc
---

# TRK-08 — SplineBoris omits species factors in physical charge and momentum

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`track_splineboris.h:69–85`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_splineboris.h#L69-L85) |

`SplineBoris` omits the particle's charge and mass ratios when converting normalized coordinates to physical charge and momentum, so non-reference species get wrong deflections and flight times. The Lorentz impulse in a uniform field scales with the charge-to-mass ratio relative to the reference species, χ, and relativistic kinematics gives field-free particles of equal normalized velocity equal arrival times. Over 0.5 m in a 0.1 T vertical field at reference momentum 1 GeV/c, a particle with twice the reference charge and the same mass (χ = 2) gets normalized horizontal momentum −0.01499 instead of −0.02998. Field-free, doubling both mass and charge gives a longitudinal displacement of −0.5000 m instead of zero. Reference-species particles are tracked correctly, and finer integration removes neither error.

??? example "Reproducer: TRK-08.py"

    [Download TRK-08.py](TRK-08.py)

    ```python
    """TRK-08: SplineBoris omits species factors in physical charge and momentum

    `SplineBoris` omits the particle's charge and mass ratios when converting normalized coordinates to
    physical charge and momentum, so non-reference species get wrong deflections and flight times. The
    Lorentz impulse in a uniform field scales with the charge-to-mass ratio relative to the reference
    species, χ, and relativistic kinematics gives field-free particles of equal normalized velocity
    equal arrival times. Over 0.5 m in a 0.1 T vertical field at reference momentum 1 GeV/c, a particle
    with twice the reference charge and the same mass (χ = 2) gets normalized horizontal momentum
    −0.01499 instead of −0.02998. Field-free, doubling both mass and charge gives a longitudinal
    displacement of −0.5000 m instead of zero. Reference-species particles are tracked correctly, and
    finer integration removes neither error.

    The script tracks particles of equal mass with χ = 0.5, 1 and 2 through a 0.5 m `SplineBoris` with a
    uniform 0.1 T vertical field and compares their horizontal momenta with the exact impulse
    −χ·B·L·c/p0c. A field-free element then tracks a reference particle and one with doubled mass and
    charge, both at the reference velocity, which must show zero longitudinal displacement. Both cases
    run with 8 and 16 integration steps, and the reference-species particles must agree.
    """

    import numpy as np
    import xtrack as xt
    from scipy.constants import c

    RTOL, ATOL = 2e-13, 2e-15


    def evaluate():
        cases = {}
        chi = np.array([0.5, 1, 2])
        for steps in (8, 16):
            element = xt.SplineBoris(length=0.5, n_steps=steps, by=xt.Spline4(0.1, 0, 0.1, 0, 0.1))
            particle = xt.Particles(p0c=1e9, delta=0, mass_ratio=1, charge_ratio=chi)
            element.track(particle)
            assert np.all(particle.state == 1)
            # Lorentz equation dp_x/ds=-q B_y, normalized by mass_ratio*p0c/c.
            expected = -chi * 0.1 * 0.5 * c / 1e9
            np.testing.assert_allclose(particle.px[1], expected[1], rtol=RTOL, atol=ATOL)
            cases[f"uniform By, {steps} steps, px for chi=0.5/1/2"] = particle.px, expected
            drift = xt.SplineBoris(length=0.5, n_steps=steps)
            particle = xt.Particles(p0c=1e9, delta=0, charge_ratio=[1, 2], mass_ratio=[1, 2])
            drift.track(particle)
            assert np.all(particle.state == 1)
            # At delta=0 both species have the reference velocity: p/m=p0/m0.
            np.testing.assert_allclose(np.asarray(particle.zeta)[0], 0, rtol=0, atol=ATOL)
            cases[f"field-free equal-chi species, {steps} steps, zeta"] = particle.zeta, np.zeros(2)
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: px=-chi*B*L*c/p0c, and equal-chi field-free species at delta=0 have zeta=0.")
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
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

