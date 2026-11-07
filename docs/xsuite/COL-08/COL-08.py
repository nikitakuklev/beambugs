"""COL-08: Loss-map initial energy double-counts preceding batches

`LossMap.add_particles` adds the energy of the whole accumulated population to `tot_energy_initial`
on every call, so a loss map built from several batches counts the earlier batches again. The
initial energy is additive: splitting an ensemble into batches or reordering them must not change
it. For seven primary protons at 3 TeV/c, the merged map reports 2.100 × 10¹³ eV, batches of 2 then
5 report 2.700 × 10¹³ eV and 5 then 2 report 3.600 × 10¹³ eV; two equal batches report 3 times one
batch's energy instead of 2. `num_initial` is correct, and loss maps filled in a single call are
unaffected.

The script builds batches of two and five primary protons with disjoint particle IDs and no losses,
and adds them to loss maps merged, as 2 then 5, as 5 then 2, and as two equal batches. It compares
the split results with the merged map, which equals the sum of the separately evaluated batch
energies, and checks that `num_initial` agrees.
"""

from _prebuilt import CTX, identity, kernels, np, report, xc, xt

identity()
line = xt.Line(
    elements=[
        xt.Drift(length=1),
        xt.LimitRect(min_x=-0.004, max_x=0.004),
        xt.Drift(length=3),
        xt.LimitRect(min_x=-0.002, max_x=0.002),
        xt.Drift(length=5),
        xt.LimitRect(min_x=-0.0007, max_x=0.0007),
        xt.Drift(length=3),
    ],
    element_names=["d0", "ap0", "d1", "ap1", "d2", "ap2", "d3"],
)
line.particle_ref = xt.Particles(p0c=2e9, pdg_id=2212)
line.build_tracker(_context=CTX)


def lossmap(part, reversed=False):
    return xc.LossMap(
        line=line,
        part=part,
        line_is_reversed=reversed,
        interpolation=False,
        correct_aperture_absorption=False,
        verbose=False,
    )


# No-loss primary batches avoid all three aperture defects. Splitting or
# permuting an ensemble cannot change its initial-energy normalization.
N1, N2 = 2, 5
p0c = 3e12
line.particle_ref = xt.Particles(p0c=p0c, pdg_id=2212)


def primary_batch(ids):
    return xt.Particles(
        _context=CTX,
        p0c=p0c,
        pdg_id=2212,
        x=np.zeros(len(ids)),
        particle_id=ids,
        parent_particle_id=ids,
    )


batch_a = primary_batch(np.arange(N1))
batch_b = primary_batch(np.arange(N1, N1 + N2))
merged = lossmap(primary_batch(np.arange(N1 + N2)))
ab = lossmap(batch_a.copy())
first_a = ab.tot_energy_initial
ab.add_particles(
    line=line,
    part=batch_b.copy(),
    interpolation=False,
    correct_aperture_absorption=False,
    verbose=False,
)
ba = lossmap(batch_b.copy())
first_b = ba.tot_energy_initial
ba.add_particles(
    line=line,
    part=batch_a.copy(),
    interpolation=False,
    correct_aperture_absorption=False,
    verbose=False,
)
print(
    f"MERGED={merged.tot_energy_initial:.17g} "
    f"SPLIT_2_THEN_5={ab.tot_energy_initial:.17g} "
    f"SPLIT_5_THEN_2={ba.tot_energy_initial:.17g}"
)
print(
    f"batch_energies_sum={first_a + first_b:.17g} "
    f"population*reference_energy={(N1 + N2) * np.hypot(p0c, batch_a.mass0):.17g} "
    "(single-batch implementation uses p0c, not energy0)"
)
assert merged.num_initial == ab.num_initial == ba.num_initial == N1 + N2
assert merged.tot_energy_initial == first_a + first_b == 7 * p0c
equal = lossmap(batch_a.copy())
equal.add_particles(
    line=line,
    part=batch_a.copy(),
    interpolation=False,
    correct_aperture_absorption=False,
    verbose=False,
)
print("EQUAL_BATCHES ratio=", equal.tot_energy_initial / first_a, "expected=2")
kernels()
report(
    {
        "split 2 then 5": (ab.tot_energy_initial, merged.tot_energy_initial),
        "split 5 then 2": (ba.tot_energy_initial, merged.tot_energy_initial),
        "equal-batch ratio": (equal.tot_energy_initial / first_a, 2.0),
    },
    rtol=0,
    atol=0,
)
