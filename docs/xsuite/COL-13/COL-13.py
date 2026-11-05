"""COL-13: Everest deposited-energy accumulators omit macroparticle weights

`EverestCollimator` adds each particle's energy loss to its deposited-energy counters without the
macroparticle `weight`, the number of physical particles a macroparticle represents. The deposited
energy is the sum of `weight` times energy loss, the weighting Xcoll's Geant4 interface applies to
the same counters. For 20 protons at 400 GeV/c through 0.6 mm of carbon-fibre composite with fixed
random seeds, weights 1 to 5 give a primary deposit of 2.865 × 10⁶ eV, the same as unit weights,
instead of 8.653 × 10⁶ eV. By source reading, the secondary-particle counter, `EverestBlock` and
`EverestCrystal` contain the same expression.

The script tracks 20 protons from x = 3 mm through the collimator (jaw 1 mm) with per-particle seeds
7001–7020, all surviving, once with unit weights and once with weights 1, 2, 3, 4, 5 repeated four
times. The per-particle energy losses are identical in both runs; the script checks that the
accumulated deposit equals their unweighted sum and compares it with the weighted sum. It requires
the installed prebuilt kernels (compilation is forbidden).
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
    print(
        f"{label}: primary_counter={primary:.17g} secondary_counter={secondary:.17g} unweighted_dE={energy_loss.sum():.17g} weighted_dE={physical:.17g}"
    )
    return recorded, physical, energy_loss


unit = run("unit-primary", np.ones(n), 1)
weighted = run("weighted-primary", 1.0 + np.arange(n) % 5, 1)
np.testing.assert_array_equal(unit[2], weighted[2])
assert unit[0] == weighted[0]
assert abs(weighted[0] - weighted[1]) > 1e6
print("IDENTICAL energy losses and accumulator with unit and nonuniform weights")
kernels()
print("DEFECT REPRODUCED")
