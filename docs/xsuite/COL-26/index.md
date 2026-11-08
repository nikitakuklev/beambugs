---
title: "COL-26: Everest clock can precede the shortest flight between sampled endpoints"
hide:
  - toc
---

# COL-26 — Everest clock can precede the shortest flight between sampled endpoints

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-06 on xcoll release [v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5) (`64e13f5`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [scamcs, multiple_coulomb_scattering.h:85–109](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/scattering_routines/everest/multiple_coulomb_scattering.h#L85-L109)<br>[lines 201–208](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/scattering_routines/everest/multiple_coulomb_scattering.h#L201-L208)<br>[EverestBlock, everest_block.h:124–147](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/beam_elements/elements_src/everest_block.h#L124-L147)<br>[drift_zeta_single](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/scattering_routines/everest/everest.h#L71-L74)<br>[jaw energy update](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/scattering_routines/everest/jaw.h#L48-L60)<br>[surviving nuclear channels](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/scattering_routines/everest/nuclear_interaction.h#L53-L95)<br>[EverestCollimator](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/beam_elements/elements_src/everest_collimator.h#L223-L225)<br>[EverestCrystal](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/beam_elements/elements_src/everest_crystal.h#L263-L265)<br>[ζ = s − β₀ct](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L334-L339)<br>[semi-continuous proton scattering](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/collimation.rst#L440-L447) |

Everest's surviving-particle clock can imply arrival before the shortest path between its sampled endpoints. The clock uses entrance and exit slopes but omits the independently sampled transverse excursion and internal collision locations. In a 0.6000 m carbon-fibre block, a 450 GeV/c proton entering at slopes (10.00, 30.00) µrad arrives at least 0.2903 fs too early; 85 of 648 survivors violate the geometric bound. High-precision endpoint geometry and a conditional diffusion calculation establish the inconsistency, although the unconditional frozen-energy mean clock is correct.

Run `python COL-26.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-26.py"

    [Download COL-26.py](COL-26.py)

    ```python
    """COL-26: Everest clock can precede the shortest flight between sampled endpoints

    Everest's surviving-particle clock can imply arrival before the shortest path between its sampled endpoints. The clock uses entrance and exit slopes but omits the independently sampled transverse excursion and internal collision locations. In a 0.6000 m carbon-fibre block, a 450 GeV/c proton entering at slopes (10.00, 30.00) µrad arrives at least 0.2903 fs too early; 85 of 648 survivors violate the geometric bound. High-precision endpoint geometry and a conditional diffusion calculation establish the inconsistency, although the unconditional frozen-energy mean clock is correct.

    Track seeded protons through carbon-fibre and elemental-carbon blocks and compare each surviving clock with a 70-digit endpoint-distance bound. Independently condition angular diffusion on displacement and exit angle, and check inactive transport and the unconditional mean.
    """

    import json
    import math

    import mpmath as mp
    from _prebuilt import CTX, identity, kernels, np, xc, xt

    L = 0.6
    C = 299792458.0
    RESULTS = []


    def main():
        identity(random_seed="12001-14048")
        failures = 0
        material = xc.materials.CarbonFibreComposite
        print("MATERIAL", material.to_dict())
        mp.mp.dps = 70
        for momentum in (450e9, 7e12):
            for a, b in ((0.0, 0.0), (10e-6, 30e-6), (100e-6, 300e-6), (0.001, 0.003)):
                n = 2048
                p = xt.Particles(_context=CTX, p0c=momentum, x=np.zeros(n), px=a, py=b)
                p._init_random_number_generator(seeds=np.arange(12001, 12001 + n, dtype=np.uint32))
                block = xc.EverestBlock(_context=CTX, length=L, material=material)
                block.track(p)
                # Nuclear absorption reorders storage: all comparisons use surviving
                # coordinates from the same slot and report the stable particle ID.
                alive = np.flatnonzero(np.asarray(p.state) > 0)
                assert len(alive) > 0
                assert np.all(np.asarray(p.delta)[alive] <= 0)
                x = np.asarray(p.x)[alive]
                y = np.asarray(p.y)[alive]
                z = np.asarray(p.zeta)[alive]
                u = (x * x + y * y) / L**2
                chord_excess = L * u / (np.sqrt(1 + u) + 1)
                deficit = chord_excess + z
                i = int(np.argmax(deficit))
                j = int(alive[i])
                # Independent 70-digit direct square root verifies the double
                # rationalization and excludes cancellation as the cause.
                precise = (
                    mp.sqrt(mp.mpf(L) ** 2 + mp.mpf(float(x[i])) ** 2 + mp.mpf(float(y[i])) ** 2)
                    - mp.mpf(L)
                    + mp.mpf(float(z[i]))
                )
                assert abs(float(precise) - deficit[i]) < 8e-21
                r = {
                    "case": "surviving endpoint flight bound",
                    "expected_max_deficit_m": 0.0,
                    "derivation": "beta0*c*dt >= endpoint chord; 70-digit direct square root",
                    "p0c_eV": momentum,
                    "slopes": [a, b],
                    "survivors": len(alive),
                    "violations_3fm": int(np.count_nonzero(deficit > 3e-15)),
                    "worst_id": int(p.particle_id[j]),
                    "x": float(x[i]),
                    "y": float(y[i]),
                    "zeta": float(z[i]),
                    "delta": float(p.delta[j]),
                    "max_deficit_m": float(deficit[i]),
                    "max_deficit_70digit": str(precise),
                    "max_deficit_fs": float(deficit[i] / (p.beta0[j] * C) * 1e15),
                }
                failures += r["violations_3fm"]
                RESULTS.append(r)
                print(json.dumps(r))
        # Exact within the paraxial Gaussian angular-diffusion model: condition a
        # Brownian angle on its final angle b and displacement I. In units t=s/L,
        # E[angle-a | I,b] = 6t(1-t)*I/L + (3t^2-2t)*b.
        # Integral conditional variance = q*L^2/15. Integrating the square gives
        # L*a^2+2*a*I+6*I^2/(5L)-I*b/5+2L*b^2/15+qL^2/15.
        # The native sampler has Var(b)=sigma^2 and
        # I=L*sigma*(z2+z1/sqrt(3))/2, so q=sigma^2/L.
        radl = float(material.radiation_length)
        momentum_GeV = 450.0
        sigma = 0.0136 / momentum_GeV * math.sqrt(L / radl) * (1 + 0.038 * math.log(L / radl))
        for a in (0.0, 30e-6, 0.003):
            z1, z2 = 1.0, 0.0
            inc = sigma * z2
            displacement = L * sigma * (z2 + z1 / math.sqrt(3)) / 2
            proxy = L / 4 * (a * a + (a + inc) ** 2)
            conditional = 0.5 * (
                L * a * a
                + 2 * a * displacement
                + 6 * displacement**2 / (5 * L)
                - displacement * inc / 5
                + 2 * L * inc**2 / 15
                + sigma**2 * L / 15
            )
            minimum = (a * L + displacement) ** 2 / (2 * L)
            r = {
                "analytic_incoming": a,
                "sigma": sigma,
                "excursion": displacement,
                "endpoint_clock_excess": proxy,
                "paraxial_chord_excess": minimum,
                "conditional_mean_excess": conditional,
                "conditional_error": conditional - proxy,
                "native_missing_z1_cross_term": a * displacement,
            }
            RESULTS.append(r)
            print(json.dumps(r))
            assert conditional >= minimum
            assert proxy < minimum
        # Attempted falsification: the endpoint clock has the correct unconditioned
        # mean in the frozen-energy diffusion model. What fails is the joint path,
        # displacement and clock law, rather than that mean or energy conservation.
        print(
            "UNCONDITIONED_MEAN_CONTROL",
            "endpoint and diffusion means both L*a^2/2 + L*sigma^2/4 per plane",
        )
        print(
            "FIRST_ORDER_COVARIANCE",
            "Cov(excess_path,excursion): diffusion=a*sigma^2*L^2/3; endpoint_clock=a*sigma^2*L^2/4; ratio=0.75",
        )
        # An elemental material with no supplied nuclear cross sections isolates MCS
        # and ionisation from the additional collision-location approximation.
        elemental = xc.Material(Z=6, A=12.011, density=1.67, name="elemental_carbon_mcs")
        n = 2048
        p = xt.Particles(_context=CTX, p0c=450e9, x=np.zeros(n), px=10e-6, py=30e-6)
        p._init_random_number_generator(seeds=np.arange(12001, 12001 + n, dtype=np.uint32))
        xc.EverestBlock(_context=CTX, length=L, material=elemental).track(p)
        assert np.all(p.state > 0)
        u = (p.x**2 + p.y**2) / L**2
        deficit = L * u / (np.sqrt(1 + u) + 1) + p.zeta
        r = {
            "control": "physical elemental carbon MCS and ionisation",
            "max_deficit_m": float(np.max(deficit)),
            "violations_3fm": int(np.count_nonzero(deficit > 3e-15)),
            "survivors": n,
        }
        RESULTS.append(r)
        print(json.dumps(r))
        p = xt.Particles(_context=CTX, p0c=450e9, px=10e-6, py=30e-6)
        xc.EverestBlock(_context=CTX, length=L, material=material, active=False).track(p)
        expected = -L * (10e-6**2 + 30e-6**2) / 2
        np.testing.assert_allclose(np.asarray(p.zeta)[0], expected, atol=3e-16, rtol=0)
        print("INACTIVE_CLOCK_CONTROL", float(np.asarray(p.zeta)[0]), "EXPECTED", expected)
        kernels()
        print("DEFECT REPRODUCED" if failures else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xcoll 0.12.5 (`64e13f596c87`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xcoll 0.12.5 (`64e13f596c87`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

