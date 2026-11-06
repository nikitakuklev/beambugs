---
title: "TRK-11: Integer factorial overflow corrupts multipole normalization"
hide:
  - toc
---

# TRK-11 — Integer factorial overflow corrupts multipole normalization

| | |
|---|---|
| Code | [ImpactX](../index.md) · Tracking |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-28 on ImpactX 26.10 (`615921c`); not reproduced on ImpactX development (`f9c2a82`), 2026-11-05 |
| Pull request | [[gh:https://github.com/BLAST-ImpactX/impactx/pull/1759]], fix |
| Source | [`Multipole.H#L54-L60`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/Multipole.H#L54-L60)<br>[`ExactMultipole.H#L307-L322`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/elements/ExactMultipole.H#L307-L322)<br>[multipole contract](https://github.com/BLAST-ImpactX/impactx/blob/615921c6867c895cf3a8fcf1d12991079d0f7d75/docs/source/usage/parameters.rst#L1162-L1170) |

`Multipole` and `ExactMultipole` compute their factorial normalization in 32-bit integers, which overflow from 13! = 6227020800 on: thin `multipole=14` and above, and thick coefficients at index 13 and above. At y = 0 a normal multipole of integrated strength K and power n (n = `multipole` − 1 when thin) must change `px` by −K·xⁿ/n! with the exact factorial; one drift–kick–drift step of the thick element has the same exact expectation. On the CPU build, order-14 normal and skew kicks are 3.223 times too large (−0.003223 instead of −0.001 at x = 0.25 m), and order-18 kicks have the wrong sign and are 1.233e6 times too large. Orders up to 13 are correct; signed overflow makes higher-order values compiler-dependent.

??? example "Reproducer: TRK-11.py"

    [Download TRK-11.py](TRK-11.py)

    ```python
    """TRK-11: Integer factorial overflow corrupts multipole normalization

    `Multipole` and `ExactMultipole` compute their factorial normalization in 32-bit integers, which
    overflow from 13! = 6227020800 on: thin `multipole=14` and above, and thick coefficients at index 13
    and above. At y = 0 a normal multipole of integrated strength K and power n (n = `multipole` − 1
    when thin) must change `px` by −K·xⁿ/n! with the exact factorial; one drift–kick–drift step of the
    thick element has the same exact expectation. On the CPU build, order-14 normal and skew kicks are
    3.223 times too large (−0.003223 instead of −0.001 at x = 0.25 m), and order-18 kicks have the wrong
    sign and are 1.233e6 times too large. Orders up to 13 are correct; signed overflow makes
    higher-order values compiler-dependent.

    The script uses a reference particle with 1000 MeV rest energy and 1000 MeV kinetic energy, and
    particles at x = 0.25 m and 0.125 m with all other coordinates zero. For powers 11, 12, 13 and 17 it
    pushes them through a thin `Multipole` and through a 0.2 m `ExactMultipole` with one second-order
    step, with a normal or skew strength chosen so that the exact kick at x = 0.25 m is 1e-3 (1e-8 for
    power 17), and compares the kicks with Python's exact factorial and with its 32-bit wrapped value.
    Accepted orders are compared with the exact oracle; explicit ValueError rejection of orders beyond the safe range counts as supported input validation. Thin order 13 and thick order 12 are controls and must be exact.
    """

    import ctypes
    import math
    import os
    from functools import partial

    for variable in (
        "NPROC",
        "PJRT_NPROC",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import numpy as np
    from impactx import ImpactX, elements

    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = False
    sim.diagnostics = False
    sim.slice_step_diagnostics = False
    sim.init_grids()


    def push(element, points):
        sim.beam.clear_particles()
        reference = sim.beam.ref
        reference.reset()
        reference.set_charge_qe(1).set_mass_MeV(1000).set_kin_energy_MeV(1000)
        zeros = np.zeros_like(points)
        sim.beam.add_n_particles(
            x=points.copy(),
            y=zeros.copy(),
            t=zeros.copy(),
            px=zeros.copy(),
            py=zeros.copy(),
            pt=zeros.copy(),
            qm=reference.qm_ratio_SI,
            bunch_charge=1e-15,
        )
        sim.lattice.clear()
        sim.lattice.append(element)
        sim.track_particles()
        return sim.beam.to_df(local=True)[["momentum_x", "momentum_y"]].to_numpy(copy=True)


    bad = False
    try:
        print(f"native C int width={ctypes.sizeof(ctypes.c_int) * 8}")
        for exponent in (11, 12, 13, 17):
            order = exponent + 1
            amplitude = 1e-3 if exponent < 17 else 1e-8
            x = 0.25
            length = 0.2
            # x = 1/4 is exact in binary; Python's factorial is integer-exact.
            integrated = math.factorial(exponent) * amplitude / x**exponent
            wrapped = ctypes.c_int(math.factorial(exponent)).value
            overflow_factor = math.factorial(exponent) / wrapped
            print(
                f"order={order} exponent={exponent} true_factorial={math.factorial(exponent)} wrapped_factorial={wrapped} ratio={overflow_factor:.15g}"
            )
            for component in ("normal", "skew"):
                normal = integrated if component == "normal" else 0.0
                skew = integrated if component == "skew" else 0.0
                thick_normal = [0.0] * exponent + [normal / length]
                thick_skew = [0.0] * exponent + [skew / length]
                models = [
                    (
                        "thin",
                        partial(elements.Multipole, multipole=order, K_normal=normal, K_skew=skew),
                    ),
                    (
                        "thick",
                        partial(
                            elements.ExactMultipole,
                            ds=length,
                            k_normal=thick_normal,
                            k_skew=thick_skew,
                            int_order=2,
                            mapsteps=1,
                            nslice=1,
                        ),
                    ),
                ]
                # With zero entrance transverse momentum, the thick element's single second-order
                # drift/kick/drift step evaluates its kick at the entrance x, so the exit momentum has
                # this exact discrete-map expectation, without treating a finite magnet as a thin limit.
                positions = np.array([x, x / 2])
                expected = (
                    np.array([-normal, skew])
                    * positions[:, None] ** exponent
                    / math.factorial(exponent)
                )
                for name, construct in models:
                    safe_order = 13 if name == "thin" else 12
                    try:
                        element = construct()
                    except ValueError as exc:
                        if order <= safe_order:
                            raise
                        print(f"{name} {component} order={order}: rejected unsupported order: {exc}")
                        continue
                    observed = push(element, positions)
                    matches = np.allclose(observed, expected, rtol=2e-14, atol=1e-23)
                    if order <= safe_order:
                        np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=1e-23)
                    elif not matches:
                        # Confirm the overflow signature independently of the verdict.
                        np.testing.assert_allclose(
                            observed, expected * overflow_factor, rtol=2e-14, atol=1e-23
                        )
                        assert not np.allclose(observed, expected, rtol=1e-12, atol=1e-23)
                        bad = True
                    index = 0 if component == "normal" else 1
                    print(
                        f"{name} {component} order={order}: expected={expected[0, index]:.15g} actual={observed[0, index]:.15g} ratio={observed[0, index] / expected[0, index]:.15g}"
                    )
    finally:
        sim.finalize()
    print("DEFECT REPRODUCED" if bad else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

