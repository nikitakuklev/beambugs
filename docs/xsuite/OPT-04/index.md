---
title: "OPT-04: Finite-difference transfer matrices reset the supplied RF clock"
hide:
  - toc
---

# OPT-04 — Finite-difference transfer matrices reset the supplied RF clock

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack main (`3cd8bfc`) |
| Source | [`get_R_matrix`, transfer_matrices.py:54–69](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/transfer_matrices.py#L54-L69)<br>[reference-field copy](https://github.com/xsuite/xpart/blob/2633e528018af2e2de24db94122fbab7956ab12e/xpart/build_particles.py#L301-L313) |

The finite-difference transfer matrix (`get_R_matrix`) and open Twiss propagation drop the supplied orbit's clock `time_s`, resetting the RF clock of an absolute-time cavity. Shifting the clock while compensating with the cavity phase leaves tracking and its derivatives unchanged, as the analytic RF focusing and central differences that keep the full particle state confirm. For a 1 GeV/c proton and a 1 MV, 100 MHz cavity with a compensated clock phase of π/3 rad, the longitudinal focusing term, the change of momentum coordinate `pzeta` per longitudinal offset `zeta`, is −1.970 × 10⁻³ m⁻¹ instead of −3.941 × 10⁻³ m⁻¹, and open Twiss generates `pzeta` = −1.188 × 10⁻³ instead of zero. Zero-clock controls pass.

??? example "Reproducer: OPT-04.py"

    [Download OPT-04.py](OPT-04.py)

    ```python
    """OPT-04: Finite-difference transfer matrices reset the supplied RF clock

    The finite-difference transfer matrix (`get_R_matrix`) and open Twiss propagation drop the supplied
    orbit's clock `time_s`, resetting the RF clock of an absolute-time cavity. Shifting the clock while
    compensating with the cavity phase leaves tracking and its derivatives unchanged, as the analytic RF
    focusing and central differences that keep the full particle state confirm. For a 1 GeV/c proton and
    a 1 MV, 100 MHz cavity with a compensated clock phase of π/3 rad, the longitudinal focusing term,
    the change of momentum coordinate `pzeta` per longitudinal offset `zeta`, is −1.970 × 10⁻³ m⁻¹
    instead of −3.941 × 10⁻³ m⁻¹, and open Twiss generates `pzeta` = −1.188 × 10⁻³ instead of zero.
    Zero-clock controls pass.

    The script builds a ring of one absolute-time `Cavity` (1 MV, 100 MHz, phase equal to minus the
    clock phase) and a drift of one RF wavelength, and checks that a 1 GeV/c proton orbit with clock
    phase 0 or π/3 closes. It compares the `get_R_matrix` element ∂`pzeta`/∂`zeta` with the analytic
    −(V/p0c)·2πf/(c·β₀²), where β₀ is the reference speed over c, and with central differences of
    tracked copies of the full orbit particle, and checks that open Twiss started from the orbit keeps
    `pzeta` = 0. The zero-clock case is the control.
    """

    import numpy as np
    import xtrack as xt
    from scipy.constants import c

    RTOL, ATOL = 2e-9, 2e-12


    def evaluate():
        voltage, frequency, p0c = 1e6, 1e8, 1e9
        cases = {}
        for clock_phase in (0.0, np.pi / 3):
            # Compensating phase puts the supplied orbit at a zero-voltage crossing.
            reference = xt.Particles(p0c=p0c, time_s=clock_phase / (2 * np.pi * frequency))
            beta0 = float(reference.beta0[0])
            # One ring revolution is exactly one RF period, so this is a periodic orbit.
            line = xt.Line(
                elements={
                    "rf": xt.Cavity(
                        voltage=voltage, frequency=frequency, phase=-clock_phase, absolute_time=True
                    ),
                    "d": xt.Drift(length=beta0 * c / frequency),
                }
            )
            line.particle_ref = reference.copy()
            line.build_tracker()
            orbit = reference.copy()
            line.track(orbit)
            np.testing.assert_allclose([orbit.zeta[0], orbit.pzeta[0]], [0, 0], rtol=0, atol=1e-14)
            expected = -voltage / p0c * (2 * np.pi * frequency / c) / beta0**2
            # An independent central difference copies the entire reference state.
            h = 2e-6
            samples = []
            for zeta in (-h, h):
                probe = reference.copy()
                probe.zeta = zeta
                line.track(probe)
                samples.append(float(probe.pzeta[0]))
            direct = (samples[1] - samples[0]) / (2 * h)
            np.testing.assert_allclose(direct, expected, rtol=RTOL, atol=ATOL)
            observed = line.get_R_matrix(particle_on_co=reference, steps={"dzeta": h})["R_matrix"][5, 4]
            if clock_phase == 0:
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
            cases[f"clock phase={clock_phase}, R[pzeta,zeta]"] = observed, expected
            init = xt.TwissInit(particle_on_co=reference.copy(), W_matrix=np.eye(6), element_name="rf")
            tw = line.twiss(init=init, chrom=False)
            if clock_phase == 0:
                np.testing.assert_allclose(tw.pzeta, 0, rtol=0, atol=1e-14)
            cases[f"clock phase={clock_phase}, open Twiss pzeta"] = tw.pzeta, np.zeros(3)
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: -V/p0c*2*pi*f/(c*beta0**2), invariant under a simultaneous clock/phase shift.")
        print("The orbit closes; independent tracking differences preserve its clock.")
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

