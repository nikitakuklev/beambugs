---
title: "COL-15: Loss-map export drops deposits without absorbed particles"
hide:
  - toc
---

# COL-15 — Loss-map export drops deposits without absorbed particles

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Failure | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xcoll release [v0.12.5](https://github.com/xsuite/xcoll/tree/v0.12.5) (`64e13f5`) |
| Source | [`LossMap.lossmap`, lossmap.py:161–172](https://github.com/xsuite/xcoll/blob/64e13f596c879cb28edded913e3c97261a3401f2/xcoll/lossmap.py#L161-L172) |

`LossMap` exports a collimator row only when particles were absorbed in it (`summary.n` > 0), so energy deposited by particles that leave the collimator is lost through `to_json` and `from_json`. The summary holds that energy before export, and a round trip must preserve it. For 20 protons at 400 GeV/c through 0.6 mm of carbon-fibre composite with no absorption, the collimator's 2.865 × 10⁶ eV becomes 0 after export and reload. The restored map still compares equal to the original, because equality compares the same filtered export. Saved loss maps miss the deposits of every collimator in which no particle was absorbed.

Run `python COL-15.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-15.py"

    [Download COL-15.py](COL-15.py)

    ```python
    """COL-15: Loss-map export drops deposits without absorbed particles

    `LossMap` exports a collimator row only when particles were absorbed in it (`summary.n` > 0), so
    energy deposited by particles that leave the collimator is lost through `to_json` and `from_json`.
    The summary holds that energy before export, and a round trip must preserve it. For 20 protons at
    400 GeV/c through 0.6 mm of carbon-fibre composite with no absorption, the collimator's 2.865 × 10⁶
    eV becomes 0 after export and reload. The restored map still compares equal to the original, because
    equality compares the same filtered export. Saved loss maps miss the deposits of every collimator in
    which no particle was absorbed.

    The script tracks 20 unit-weight protons from x = 3 mm through the collimator (jaw 1 mm, seeds
    7001–7020), all surviving, and checks the in-memory collimator energy against the kernel's deposit
    counter. It then writes the loss map to JSON in a temporary directory, reloads it, and compares the
    restored collimator energy, and the equality test, with the original.
    """

    import tempfile
    from pathlib import Path

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
        print(
            f"{label}: primary_counter={primary:.17g} secondary_counter={secondary:.17g} unweighted_dE={energy_loss.sum():.17g} weighted_dE={physical:.17g}"
        )
        print(f"{label}: summary.n=0 summary.e={lossmap.tot_energy_collimator:.17g}")
        file = Path(tempfile.mkdtemp()) / "COL-15-deposits.json"
        lossmap.to_json(file)
        restored = xc.LossMap.from_json(file, verbose=False)
        print(
            f"{label}: JSON rows={len(lossmap.lossmap['collimator']['name'])} restored.e={restored.tot_energy_collimator:.17g} LossMap_equality={restored == lossmap}"
        )
        assert len(lossmap.lossmap["collimator"]["name"]) == 0
        assert restored.tot_energy_collimator == 0
        # Equality itself uses the filtered export and therefore misses the loss.
        assert restored == lossmap


    run("unit-primary", np.ones(n), 1)
    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-10-28</span> | xcoll 0.12.5 (`64e13f596c87`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

