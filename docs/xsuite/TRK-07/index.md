---
title: "TRK-07: Optimization disables radiation in curved-body RBend slices"
hide:
  - toc
---

# TRK-07 — Optimization disables radiation in curved-body RBend slices

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P1 |
| Failure | Wrong physics |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack release [v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4) (`3cd8bfc`) |
| Source | [`ThinSliceRBend.get_equivalent_element`, slice_elements_thin.py:220–242](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/slice_elements_thin.py#L220-L242) |

`Line.optimize_for_tracking()` silently turns off the configured mean synchrotron radiation in thin slices of a curved-body `RBend`. The classical radiated power predicts a finite energy loss, the unoptimized slices reproduce it, and an equivalent sector `Bend` keeps it after optimization. For a 3 GeV/c electron in a rectangular bend with 1 m arc length, bend angle 0.01 rad and `Teapot(1)` slicing, optimization changes the energy loss from 114.0 eV to zero. The `straight-body` model bypasses this conversion.

Run `python TRK-07.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-07.py"

    [Download TRK-07.py](TRK-07.py)

    ```python
    """TRK-07: Optimization disables radiation in curved-body RBend slices

    `Line.optimize_for_tracking()` silently turns off the configured mean synchrotron radiation in thin
    slices of a curved-body `RBend`. The classical radiated power predicts a finite energy loss, the
    unoptimized slices reproduce it, and an equivalent sector `Bend` keeps it after optimization. For a
    3 GeV/c electron in a rectangular bend with 1 m arc length, bend angle 0.01 rad and `Teapot(1)`
    slicing, optimization changes the energy loss from 114.0 eV to zero. The `straight-body` model
    bypasses this conversion.

    The script builds a sector `Bend` and an `RBend` of equal arc length and angle, slices each with
    `Teapot(1)`, enables mean radiation and tracks the reference electron before and after optimization.
    The loss before optimization must match the classical radiated energy U, applied as in the mean-kick
    model by scaling the momentum by 1 − U/E₀ with E₀ the reference energy, and optimization must leave
    the particle's energy unchanged. The sector bend is the control; the `RBend` shows the bug.
    """

    import numpy as np
    import xtrack as xt

    RTOL, ATOL = 2e-13, 2e-15


    def evaluate():
        p0c, length, angle = 3e9, 1.0, 0.01
        cases = {}
        for cls, kwargs in (
            (xt.Bend, {"length": length}),
            (xt.RBend, {"length_straight": length * np.sinc(angle / (2 * np.pi))}),
        ):
            line = xt.Line(elements={"bend": cls(angle=angle, **kwargs)})
            line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=-1, p0c=p0c)
            line.slice_thick_elements(slicing_strategies=[xt.Strategy(xt.Teapot(1))])
            line.configure_radiation(model="mean")
            before = line.particle_ref.copy()
            line.track(before)
            loss_before = -float(before.ptau[0]) * p0c
            # Independent Larmor power, using the physical constants of the C model.
            # U = (2/3) r_e gamma^2 p0c^2 angle^2 / (m L). The mean-kick model scales
            # momentum by 1-U/E0 rather than subtracting U.
            mass = xt.ELECTRON_MASS_EV
            energy0 = np.hypot(p0c, mass)
            gamma0 = energy0 / mass
            classical_radius = 1.60217662e-19 / (4 * np.pi * 8.854187817620e-12 * mass)
            u_radiated = (2 / 3) * classical_radius * gamma0**2 * p0c**2 / mass * angle**2 / length
            fractional = u_radiated / energy0
            change_energy_squared = p0c**2 * fractional * (2 - fractional)
            expected_loss = change_energy_squared / (
                energy0 + np.sqrt(energy0**2 - change_energy_squared)
            )
            np.testing.assert_allclose(loss_before, expected_loss, rtol=5e-9, atol=0)
            line.optimize_for_tracking(verbose=False)
            after = line.particle_ref.copy()
            line.track(after)
            assert after.state[0] == 1
            if cls is xt.Bend:
                np.testing.assert_allclose(after.ptau, before.ptau, rtol=RTOL, atol=ATOL)
            # Compare energy coordinates to avoid cancellation in total energy.
            cases[f"{cls.__name__}, optimized ptau (same selected mean-radiation map)"] = (
                np.asarray(after.ptau),
                np.asarray(before.ptau),
            )
        return cases


    if __name__ == "__main__":
        from _common import prepare, report

        prepare(radiation="mean")
        print(
            "Expected: optimization preserves radiative ptau; independent Larmor check gives about 114.0422 eV loss."
        )
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-26</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | low | <span class="bb-nowrap">2026-10-26</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels, CPU, serial |

