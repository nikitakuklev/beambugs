---
title: "COL-16: Touschek Monte Carlo diagnostic omits sampled angular width"
hide:
  - toc
---

# COL-16 — Touschek Monte Carlo diagnostic omits sampled angular width

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields release [v0.27.7](https://github.com/xsuite/xfields/tree/v0.27.7) (`3f06c87`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`TouschekScatter` prefactor, touschek.h:374–376](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/touschek_src/touschek.h#L374-L376)<br>[angular sampling, lines 443–446](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/touschek_src/touschek.h#L443-L446) |

`TouschekScatter` samples the polar scattering angle θ uniformly between `theta_min` and `theta_max` and the azimuth over π, but always uses the Monte Carlo volume π² instead of π·(θₘₐₓ − θₘᵢₙ). With matched random samples for 1.4 GeV/c electrons, narrowing the interval around π/2 from 2⁻²⁰ to 2⁻²⁴ rad leaves the diagnostic `total_mc_rate` at 7464 s⁻¹, whereas the manual's uniform Monte Carlo estimator, which includes the sampled volume, gives 0.002266 and 0.0001416 s⁻¹, a 16-fold reduction. Returned particle weights are normalized to the Piwinski rate, so the factor cancels there; by source reading, `ignored_rate` inherits the diagnostic error. For the default interval the excess is 0.01000% by algebra.

Run `python COL-16.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-16.py"

    [Download COL-16.py](COL-16.py)

    ```python
    """COL-16: Touschek Monte Carlo diagnostic omits sampled angular width

    `TouschekScatter` samples the polar scattering angle θ uniformly between `theta_min` and `theta_max`
    and the azimuth over π, but always uses the Monte Carlo volume π² instead of π·(θₘₐₓ − θₘᵢₙ). With
    matched random samples for 1.4 GeV/c electrons, narrowing the interval around π/2 from 2⁻²⁰ to 2⁻²⁴
    rad leaves the diagnostic `total_mc_rate` at 7464 s⁻¹, whereas the manual's uniform Monte Carlo
    estimator, which includes the sampled volume, gives 0.002266 and 0.0001416 s⁻¹, a 16-fold reduction.
    Returned particle weights are normalized to the Piwinski rate, so the factor cancels there; by
    source reading, `ignored_rate` inherits the diagnostic error. For the default interval the excess is
    0.01000% by algebra.

    The script runs a `TouschekMarker` twice (2000 selected scattering events, Xtrack random generator
    seed 9823, integrated Piwinski rate 3.25 s⁻¹) with θ intervals of width 2⁻²⁰ and 2⁻²⁴ centred at
    π/2. Identical final generator states and sampled positions show that both runs use the same
    samples, over which the integrand changes by less than a computed bound, so a correct rate must
    shrink with the sampled volume by a factor 16. It also checks that the returned weights sum to the
    Piwinski rate.
    """

    import xfields as xf
    from _prebuilt import CTX, identity, kernels, np, xt

    identity(random_seed=9823)


    def run(width):
        low, high = np.pi / 2 - width / 2, np.pi / 2 + width / 2
        actual_width = high - low
        marker = xf.TouschekMarker(
            _context=CTX,
            p0c=1.4e9,
            bunch_intensity=2.3e10,
            betx=8.0,
            bety=12.0,
            gemitt_x=2e-8,
            gemitt_y=3e-9,
            sigma_z=0.006,
            sigma_delta=1.5e-4,
            delta_neg=-0.001,
            delta_pos=0.001,
            n_scattering_events=2000,
            nx=5,
            ny=5,
            nz=5,
            theta_min=low,
            theta_max=high,
            integrated_piwinski_rate=3.25,
            rng="xtrack",
            weight_retention_fraction=1.0,
        )
        carrier = xt.Particles(_context=CTX, p0c=marker.p0c, mass0=xt.ELECTRON_MASS_EV, q0=-1)
        carrier._init_random_number_generator(seeds=np.array([9823], dtype=np.uint32))
        generated = marker._generate(particle_ref=carrier, at_element=0, s=0, rng_particle=carrier)
        n = len(marker.theta_log)
        assert n == 2000
        rate = float(marker.total_mc_rate)
        weights = np.array(generated.weight[:n])
        np.testing.assert_allclose(weights.sum(), 3.25, rtol=2e-15, atol=0)
        print(
            f"width={actual_width:.17g} total_mc_rate={rate:.17g} volume_corrected_rate={rate * actual_width / np.pi:.17g} returned_weight_sum={weights.sum():.17g}"
        )
        rng_state = np.array([getattr(carrier, f"_rng_s{i}")[0] for i in range(1, 5)])
        positions = np.array([np.array(getattr(generated, coord)[:n]) for coord in ("x", "y", "zeta")])
        return actual_width, rate, rng_state, positions, weights


    wide = run(2.0**-20)
    narrow = run(2.0**-24)
    # Same final RNG state and identical sampled positions demonstrate identical
    # event selection and trial count; the angular integrand is the only change.
    np.testing.assert_array_equal(wide[2], narrow[2])
    np.testing.assert_array_equal(wide[3], narrow[3])
    # At theta=pi/2+e, sin(theta)*moeller(beta,theta) is even in e.
    # Its relative quadratic coefficient is (4.5*A+1.5)/(A+5) < 4.5,
    # A=(1+1/beta**2)**2 >= 4. A bound 6*e**2 plus roundoff is conservative.
    bound = 6 * (wide[0] / 2) ** 2 + 100 * np.finfo(float).eps
    ratio = narrow[1] / wide[1]
    np.testing.assert_allclose(ratio, 1.0, rtol=0, atol=bound)
    corrected_ratio = ratio * narrow[0] / wide[0]
    np.testing.assert_allclose(corrected_ratio, 1 / 16, rtol=0, atol=bound / 16)
    assert abs(ratio - 1 / 16) > 0.9
    print(
        f"MATCHED_SAMPLES actual_rate_ratio={ratio:.17g} angular_volume_ratio={narrow[0] / wide[0]:.17g} corrected_rate_ratio={corrected_ratio:.17g} integrand_bound={bound:.3g}"
    )
    print("DEFAULT_RATE_RELATIVE_EXCESS", 1 / 0.9999 - 1)
    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

