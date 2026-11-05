---
title: "TRK-10: Linear dipole-edge finite-gap focusing has wrong species scaling"
hide:
  - toc
---

# TRK-10 — Linear dipole-edge finite-gap focusing has wrong species scaling

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`DipoleEdge._update_r21_r43`, dipole_edge.py:126–134](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/dipole_edge.py#L126-L134)<br>[`DipoleEdgeLinear_single_particle`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_dipole_edge_linear.h#L30-L38) |

`DipoleEdge(model='linear')` applies the particle's charge-to-mass ratio relative to the reference species, χ, to its vertical edge kick but not inside the finite-gap angle correction, so non-reference species get wrong vertical focusing. The analytic fringe formula, and the equivalence between scaling the charge and scaling the field, require the leading zero-face-angle kick to scale with charge squared at fixed mass and momentum. With dipole strength `k` = 0.1 m⁻¹, half-gap `hgap` = 0.02 m, fringe integral `fint` = 0.5, zero face angle and y = 10 µm, a 1 GeV/c particle with twice the reference charge gets normalized vertical momentum 4.000 × 10⁻⁹ instead of 8.000 × 10⁻⁹. Reference-species and zero-gap edges are correct, and the `full` edge model scales correctly.

??? example "Reproducer: TRK-10.py"

    [Download TRK-10.py](TRK-10.py)

    ```python
    """TRK-10: Linear dipole-edge finite-gap focusing has wrong species scaling

    `DipoleEdge(model='linear')` applies the particle's charge-to-mass ratio relative to the reference
    species, χ, to its vertical edge kick but not inside the finite-gap angle correction, so
    non-reference species get wrong vertical focusing. The analytic fringe formula, and the equivalence
    between scaling the charge and scaling the field, require the leading zero-face-angle kick to scale
    with charge squared at fixed mass and momentum. With dipole strength `k` = 0.1 m⁻¹, half-gap `hgap`
    = 0.02 m, fringe integral `fint` = 0.5, zero face angle and y = 10 µm, a 1 GeV/c particle with twice
    the reference charge gets normalized vertical momentum 4.000 × 10⁻⁹ instead of 8.000 × 10⁻⁹.
    Reference-species and zero-gap edges are correct, and the `full` edge model scales correctly.

    The script tracks particles of equal mass and momentum with χ = 0.5, 1 and 2 from y = 10 µm through
    the linear edge and compares their vertical momenta with the model's own formula evaluated with the
    scaled field, χk·tan(2χk·hgap·fint)·y. The χ = 1 particle must agree, and a reference particle
    through an edge with `k` scaled by χ must reproduce each expected value. As a further control, the
    `full` edge model must give the same kick for scaled charge as for scaled field.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 2e-13, 2e-20


    def evaluate():
        k, gap, fint, y = 0.1, 0.02, 0.5, 1e-5
        chi = np.array([0.5, 1, 2])
        expected = chi * k * np.tan(2 * chi * k * gap * fint) * y
        particle = xt.Particles(p0c=1e9, y=y, delta=0, mass_ratio=1, charge_ratio=chi)
        xt.DipoleEdge(k=k, e1=0, hgap=gap, fint=fint, model="linear").track(particle)
        assert np.all(particle.state == 1)
        np.testing.assert_allclose(particle.py[1], expected[1], rtol=RTOL, atol=ATOL)
        for ratio, target in zip(chi, expected):
            reference = xt.Particles(p0c=1e9, y=y)
            xt.DipoleEdge(k=ratio * k, e1=0, hgap=gap, fint=fint, model="linear").track(reference)
            np.testing.assert_allclose(reference.py[0], target, rtol=RTOL, atol=ATOL)
            # The full edge obeys the same charge/field equivalence, independently
            # of the small nonlinear correction that distinguishes it from linear.
            mixed = xt.Particles(p0c=1e9, y=y, delta=0, mass_ratio=1, charge_ratio=ratio)
            scaled = xt.Particles(p0c=1e9, y=y)
            xt.DipoleEdge(k=k, e1=0, hgap=gap, fint=fint, model="full").track(mixed)
            xt.DipoleEdge(k=ratio * k, e1=0, hgap=gap, fint=fint, model="full").track(scaled)
            assert mixed.state[0] == scaled.state[0] == 1
            np.testing.assert_allclose(mixed.py, scaled.py, rtol=RTOL, atol=ATOL)
        return {"linear edge py for chi=0.5/1/2": (particle.py, expected)}


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print(
            "Expected at zero face angle: py=chi*k*tan(2*chi*k*hgap*fint)*y, quadratic in field at leading order."
        )
        print(
            "Independent invariant: multiplying particle charge equals multiplying the magnetic field."
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
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

