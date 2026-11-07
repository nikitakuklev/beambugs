---
title: "API-04: before_slice length changes retain stale collective-kick lengths"
hide:
  - toc
---

# API-04 — `before_slice` length changes retain stale collective-kick lengths

| | |
|---|---|
| Code | [ImpactX](../index.md) · Python API |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on ImpactX commit [`fa0a43b`](https://github.com/BLAST-ImpactX/impactx/commit/fa0a43b9bb07), committed 2026-10-28 |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`common.H#L187-L231`](https://github.com/BLAST-ImpactX/impactx/blob/b701a6e9d1f9e30cefb9a3e18e82ae7e02b9d943/src/tracking/common.H#L187-L231) |

If a `before_slice` hook changes the length `ds` of the upcoming element, ImpactX transports the element over the new length but applies its space-charge kick for the length cached before the hook. The hook contract allows modifying the upcoming element, the same change made in `before_element` gives the correct result, and independently composed drift and kick matrices reproduce both outcomes. For a 10 MeV, 0.1 A proton envelope with 1 mm rms sizes and 2D space charge, retuning a drift from 0.1 to 0.2 m halves the kick: the normalized-momentum variance ⟨pₓ²⟩ ends at 1.270×10⁻⁸ instead of 2.058×10⁻⁸ with Strang splitting. Retuning from 0.4 m doubles the kick and from zero length omits it, in first-order and Strang splitting; particle tracking shares this code.

Run `python API-04.py`.

??? example "Reproducer: API-04.py"

    [Download API-04.py](API-04.py)

    ```python
    """API-04: `before_slice` length changes retain stale collective-kick lengths

    If a `before_slice` hook changes the length `ds` of the upcoming element, ImpactX transports the
    element over the new length but applies its space-charge kick for the length cached before the hook.
    The hook contract allows modifying the upcoming element, the same change made in `before_element`
    gives the correct result, and independently composed drift and kick matrices reproduce both
    outcomes. For a 10 MeV, 0.1 A proton envelope with 1 mm rms sizes and 2D space charge, retuning a
    drift from 0.1 to 0.2 m halves the kick: the normalized-momentum variance ⟨pₓ²⟩ ends at 1.270×10⁻⁸
    instead of 2.058×10⁻⁸ with Strang splitting. Retuning from 0.4 m doubles the kick and from zero
    length omits it, in first-order and Strang splitting; particle tracking shares this code.

    The script tracks a 10 MeV, 0.1 A proton envelope (rms sizes 1 mm, normalized-momentum rms widths
    10⁻⁴) with 2D space charge through one single-slice drift whose `before_slice` hook sets `ds` = 0.2
    m, starting from 0.1, 0.4 and 0 m, in first-order and Strang splitting. It compares the full
    covariance matrix with drift and kick matrices composed from the paraxial uniform-ellipse kick, for
    both the correct and the stale kick length, and infers the ratio of applied to correct kick. A
    static 0.2 m drift and the same change made in `before_element` are the controls.
    """

    import os

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "NPROC",
        "PJRT_NPROC",
    ):
        os.environ[name] = "1"

    from math import pi

    import numpy as np
    from impactx import ImpactX, RefPart, distribution, elements

    sim = ImpactX()
    sim.particle_shape = 1
    sim.space_charge = "2D"
    sim.diagnostics = False
    sim.verbose = 0
    sim.init_grids()
    ref = RefPart().set_species("proton").set_kin_energy_MeV(10)
    beam_current = 0.1
    dist = distribution.KVdist(
        lambdaX=1e-3,
        lambdaY=1e-3,
        lambdaT=1e-2,
        lambdaPx=1e-4,
        lambdaPy=1e-4,
        lambdaPt=1e-4,
    )
    # The paraxial space-charge equation for a uniform ellipse with semiaxes
    # 2*sigma_x, 2*sigma_y gives px' = px + kappa_x*x, where
    # kappa_x = K*ds / (2*sigma_x*(sigma_x+sigma_y)),
    # K = 2*I/[I_A*(beta*gamma)^3], I_A = 4*pi*epsilon0*m*c^3/|q|.
    alfven = 4 * pi * 8.8541878188e-12 * ref.mass * 299792458**3 / abs(ref.charge)
    perveance = 2 * beam_current / (alfven * ref.beta_gamma**3)
    print(f"I_A={alfven:.16g} A K={perveance:.16g}")


    def drift(length):
        matrix = np.eye(6)
        matrix[0, 1] = matrix[2, 3] = length
        matrix[4, 5] = length / ref.beta_gamma**2
        return matrix


    def compose(cov, transport_length, kick_length, strang):
        d = drift(transport_length / 2 if strang else transport_length)
        if strang:
            cov = d @ cov @ d.T
        sx, sy = np.sqrt(cov[0, 0]), np.sqrt(cov[2, 2])
        k = np.eye(6)
        k[1, 0] = perveance * kick_length / (2 * sx * (sx + sy))
        k[3, 2] = perveance * kick_length / (2 * sy * (sx + sy))
        cov = k @ cov @ k.T
        return d @ cov @ d.T


    try:
        for strang in (False, True):
            sim.strang_split = strang
            expected = None
            for event, initial_length in (
                (None, 0.2),
                ("before_element", 0.1),
                ("before_slice", 0.1),
                ("before_slice", 0.4),
                ("before_slice", 0.0),
            ):
                sim.hook["before_element"] = lambda _: None
                sim.hook["before_slice"] = lambda _: None
                sim.init_envelope(ref, dist, beam_current)
                initial = sim.envelope.envelope.to_numpy().copy()
                sim.lattice.clear()
                sim.lattice.append(elements.Drift(ds=initial_length, nslice=1))
                if event:

                    def retune(s):
                        s.tracking_element.ds = 0.2

                    sim.hook[event] = retune
                sim.track_envelope()
                observed = sim.envelope.envelope.to_numpy().copy()
                target = compose(initial, 0.2, 0.2, strang)
                cached_length = initial_length if event == "before_slice" else 0.2
                stale_prediction = compose(initial, 0.2, cached_length, strang)
                np.testing.assert_allclose(observed, stale_prediction, rtol=7e-15, atol=1e-24)
                if event != "before_slice":
                    np.testing.assert_allclose(observed, target, rtol=7e-15, atol=1e-24)
                else:
                    assert np.max(abs(observed - target)) > 1e-10
                if expected is None:
                    expected = observed
                # Undo the variance relation d<px^2> = 2*k*<x px> + k^2*<x^2>.
                # The half drift in Strang splitting creates a nonzero <x px>.
                mid = initial.copy()
                if strang:
                    half = drift(0.1)
                    mid = half @ mid @ half.T

                def infer_k(cov, mid):
                    delta = cov[1, 1] - mid[1, 1]
                    if delta == 0:
                        return 0.0
                    return delta / (np.sqrt(mid[0, 1] ** 2 + mid[0, 0] * delta) + mid[0, 1])

                ratio = infer_k(observed, mid) / infer_k(expected, mid)
                np.testing.assert_allclose(ratio, cached_length / 0.2, rtol=0, atol=8e-15)
                print(
                    f"strang={strang} hook={event} initial_ds={initial_length:g}: Cpxpx={observed[1, 1]:.15g} correct={target[1, 1]:.15g} kick_ratio={ratio:.15g} analytic_stale_max_error={np.max(abs(observed - stale_prediction)):.3g}"
                )
    finally:
        sim.finalize()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | 26.10 (`615921c6867c`), impactx-noacc wheel, CPU, double, serial |

