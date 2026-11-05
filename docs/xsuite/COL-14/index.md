---
title: "COL-14: Loss-map primary energy uses secondary deposits"
hide:
  - toc
---

# COL-14 — Loss-map primary energy uses secondary deposits

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xcoll main (`64e13f5`) |
| Source | [`LossMap._make_coll_summary`, lossmap.py:698–701](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/lossmap.py#L698-L701) |

The collimator summary of `LossMap` computes the primary energy `e_prim` from the absorbed primary energy plus the secondary-particle deposit counter, instead of the primary one. The Everest kernel accumulates primary and secondary deposits separately, so without absorption `e_prim` must equal the primary counter. For 20 protons at 400 GeV/c through 0.6 mm of carbon-fibre composite without absorption, an all-primary batch reports `e_prim` = 0 instead of 2.865 × 10⁶ eV, and an all-secondary batch 2.865 × 10⁶ eV instead of 0. The total collimator energy is correct; its split into primary and secondary deposits is wrong.

Download these shared helpers into the same directory as the reproducer scripts:

- [Download _prebuilt.py](../helpers/_prebuilt.py) · [View source](../helpers/index.md#_prebuilt)
- [Download _common.py](../helpers/_common.py) · [View source](../helpers/index.md#_common)

??? example "Reproducer: COL-14.py"

    [Download COL-14.py](COL-14.py)

    ```python
    """COL-14: Loss-map primary energy uses secondary deposits

    The collimator summary of `LossMap` computes the primary energy `e_prim` from the absorbed primary
    energy plus the secondary-particle deposit counter, instead of the primary one. The Everest kernel
    accumulates primary and secondary deposits separately, so without absorption `e_prim` must equal the
    primary counter. For 20 protons at 400 GeV/c through 0.6 mm of carbon-fibre composite without
    absorption, an all-primary batch reports `e_prim` = 0 instead of 2.865 × 10⁶ eV, and an
    all-secondary batch 2.865 × 10⁶ eV instead of 0. The total collimator energy is correct; its split
    into primary and secondary deposits is wrong.

    The script tracks two unit-weight batches of 20 protons from x = 3 mm through the collimator (jaw 1
    mm, seeds 7001–7020), all surviving: one batch starts as primary particles, the other in state
    `SECONDARY_PARTICLE`. It reads the kernel's two deposit counters, then compares the loss map's total
    and primary collimator energy with them. It requires the installed prebuilt kernels (compilation is
    forbidden).
    """

    from _prebuilt import CTX, identity, kernels, np, xc, xt

    identity(random_seed="7001..7020")

    n = 20
    p0c = 4e11
    seeds = np.arange(7001, 7001 + n, dtype=np.uint32)


    def run(label, weights, state):
        coll = xc.EverestCollimator(
            _context=CTX,
            length=0.0006,
            jaw=0.001,
            material=xc.materials.CarbonFibreComposite,
            mark_scattered_particles=True,
        )
        line = xt.Line(elements={"carbon": coll, "drift": xt.Drift(length=0.2)})
        line.particle_ref = xt.Particles(_context=CTX, p0c=p0c, pdg_id_0=2212)
        line.build_tracker(_context=CTX)
        particles = xt.Particles(
            _context=CTX, p0c=p0c, pdg_id_0=2212, x=np.full(n, 0.003), weight=weights, state=state
        )
        particles._init_random_number_generator(seeds=seeds)
        initial = np.array(particles.energy)
        particles.reorganize()
        line.xcoll.scattering.enable()
        line.track(particles, num_turns=1)
        particles.sort(interleave_lost_particles=True)
        np.testing.assert_array_equal(particles.particle_id, np.arange(n))
        assert np.all(particles.state > 0), "Need a survival-only case to isolate deposition"
        energy_loss = initial - np.array(particles.energy)
        assert np.all(energy_loss > 0)
        # Xobjects generates the accumulator fields at class creation.
        primary = float(coll._acc_ionisation_loss)  # ty: ignore[unresolved-attribute]
        secondary = float(coll._acc_ionisation_loss_sec)  # ty: ignore[unresolved-attribute]
        recorded = primary + secondary
        physical = np.dot(energy_loss, np.array(particles.weight))
        # At these energies one energy subtraction has O(1e-4 eV) roundoff.
        # Use a conservative bound from the floating-point spacing, not a percent tolerance.
        atol = n * 4 * np.spacing(p0c)
        np.testing.assert_allclose(recorded, energy_loss.sum(), rtol=0, atol=atol)
        lossmap = xc.LossMap(
            line=line,
            part=particles,
            weights=np.array(particles.weight),
            interpolation=False,
            correct_aperture_absorption=False,
            verbose=False,
        )
        assert lossmap.summary.n.sum() == 0
        np.testing.assert_allclose(lossmap.tot_energy_collimator, recorded, rtol=0, atol=0)
        if state == 1:
            assert secondary == 0 and primary > 0
            assert lossmap.tot_primary_energy_collimator == 0
        else:
            assert primary == 0 and secondary > 0
            np.testing.assert_allclose(lossmap.tot_primary_energy_collimator, secondary, rtol=0, atol=0)
        print(
            f"{label}: primary_counter={primary:.17g} secondary_counter={secondary:.17g} unweighted_dE={energy_loss.sum():.17g} weighted_dE={physical:.17g}"
        )
        print(
            f"{label}: summary.n=0 summary.e={lossmap.tot_energy_collimator:.17g} summary.e_prim={lossmap.tot_primary_energy_collimator:.17g} expected_e_prim={(physical if state == 1 else 0):.17g}"
        )


    run("unit-primary", np.ones(n), 1)
    # Xcoll creates particle-state constants from its C header at import time.
    run("unit-secondary", np.ones(n), xc.constants.SECONDARY_PARTICLE)  # ty: ignore[unresolved-attribute]
    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

