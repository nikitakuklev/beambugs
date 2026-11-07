"""COL-22: Compound atom density counts formula units instead of atoms

Defining a compound by `n_atoms` makes `atoms_per_volume` divide by the mass of a formula unit without restoring the atoms per formula unit. Equivalent mass-fraction input gives the correct atom density. Water at 1.000 g/cm³ reports 3.343 × 10²⁸ rather than 1.003 × 10²⁹ atoms/m³. With explicit per-atom nuclear parameters, a 450.0 GeV/c silicon carbide example has an Everest nuclear interaction length of 0.4153 m instead of 0.2077 m. Multiplying all stoichiometric counts by two doubles this erroneous interaction length again.

The script counts each elemental population with a 50-digit mass balance and compares equivalent water and silicon carbide inputs. Sixteen seeded protons traverse a native EverestBlock with explicit per-atom nuclear parameters; an independent scalar RNG and elemental collision-rate sum predict every first nuclear flight. It uses the installed serial CPU prebuilt kernels with compilation forbidden.
"""

import math
from decimal import Decimal, localcontext

from _prebuilt import CTX, identity, kernels, np, xc, xt

NA = Decimal("6.02214076e23")
RTOL = 5e-15
matches = []
np.set_printoptions(precision=17, linewidth=10000)


def compare(label, observed, expected, derivation, *, control=False, rtol=RTOL, atol=0):
    observed, expected = np.asarray(observed), np.asarray(expected)
    passed = bool(np.allclose(observed, expected, rtol=rtol, atol=atol))
    if control:
        np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)
    print(f"{label}: observed={observed} expected={expected} derivation={derivation} "
          f"rtol={rtol} atol={atol} match={passed}")
    matches.append(passed)


def number_densities(rho, masses, counts):
    # One m³ contains rho*10^6 grams; each mole of formula units contains
    # n_i moles of constituent i. Sum these populations to count individual atoms.
    with localcontext() as ctx:
        ctx.prec = 50
        denominator = sum(Decimal(str(a)) * Decimal(str(n)) for a, n in zip(masses, counts))
        return np.array([float(Decimal(str(rho)) * Decimal(10)**6 * NA * Decimal(str(n))
                               / denominator) for n in counts])


def first_uniform(state):
    # Scalar integer evaluation of the specified combined Tausworthe/LCG stream,
    # independent of the native RNG kernel. The 32-bit mask models unsigned wrap.
    mask = 0xffffffff

    def taus(s, a, b, c, d):
        return (((s & c) << d) & mask) ^ ((((s << a) & mask) ^ s) >> b)

    s1, s2, s3, s4 = state
    s1 = taus(s1, 13, 19, 4294967294, 12)
    s2 = taus(s2, 2, 25, 4294967288, 4)
    s3 = taus(s3, 3, 11, 4294967280, 17)
    s4 = (1664525 * s4 + 1013904223) & mask
    return (s1 ^ s2 ^ s3 ^ s4) / 4294967296.0


def main():
    identity(random_seed="5101–5116")
    water_expected = number_densities(1.0, [1.008, 15.999], [2, 1]).sum()
    waters = []
    for scale in (1, 2, 0.5, 1 / 3):
        water = xc.Material(components=["H", "O"], n_atoms=[2 * scale, scale],
                            density=1.0, name=f"Water{scale}")
        compare(f"water counts scale={scale:g} atoms/m3", water.atoms_per_volume,
                water_expected, "rho*1e6*NA*sum(n_i)/sum(n_i*A_i)", control=scale == 1 / 3)
        compare(f"water scale={scale:g} cached/public", water._atoms_per_volume,
                water.atoms_per_volume, "cached field equals public density", control=True, rtol=0)
        waters.append(water)
    weights = np.array([2 * 1.008, 15.999]) / 18.015
    for mode, values in (("mass_fractions", weights), ("molar_fractions", [2.0, 1.0]),
                         ("atomic_fractions", [2.0, 1.0])):
        water = xc.Material(components=["H", "O"], density=1.0, name="Water_" + mode, **{mode: values})
        compare("water " + mode + " atoms/m3", water.atoms_per_volume, water_expected,
                "independent elemental mass balance", control=True)
        for field in ("electron_density", "radiation_length"):
            compare("water " + mode + " " + field, getattr(water, field), getattr(waters[0], field),
                    "equivalent composition preserves this quantity", control=True)
    nested = xc.Material(components=[xc.materials.Water, xc.materials.Silica],
                         volume_fractions=[0.5, 0.5], density=1.46, name="NestedMixture")
    nested_expected = float(NA) * 1.46e6 * sum(w / el.A for w, el in
                                              zip(nested.mass_fractions, nested.components))
    compare("nested volume mixture atoms/m3", nested.atoms_per_volume, nested_expected,
            "rho*1e6*NA*sum(w_i/A_i) after elemental resolution", control=True)
    for element in (xc.materials.Silicon, xc.materials.Carbon):
        compare(element.name + " elemental atoms/m3", element.atoms_per_volume,
                float(NA) * element.density * 1e6 / element.A, "rho*1e6*NA/A", control=True)

    si, carbon = xc.materials.Silicon, xc.materials.CarbonFibreComposite
    np.testing.assert_array_equal([si.A, carbon.A], [28.085, 12.011])
    # Fixed per-atom reference cross sections and nuclear parameters are the
    # atomic-fraction averages of the native silicon/carbon-fibre-carbon tables.
    sigma_ref = np.array([0.5005, 0.331, 0.0, 0.0, 0.0, 0.000233])
    compare("SiC per-atom reference cross sections/barn", sigma_ref,
            (si.cross_section + carbon.cross_section) / 2, "equal atomic fractions", control=True)
    params = dict(density=3.21, nuclear_radius=0.3455, nuclear_elastic_slope=95.07,
                  hcut=0.02, cross_section=sigma_ref)
    forms = [("counts_1_1", {"n_atoms": [1, 1]}), ("counts_2_2", {"n_atoms": [2, 2]}),
             ("counts_half_half", {"n_atoms": [0.5, 0.5]}),
             ("mass_fractions", {"mass_fractions": [28.085 / 40.096, 12.011 / 40.096]})]
    initial = xt.Particles(_context=CTX, p0c=450e9, x=np.zeros(16), particle_id=np.arange(16))
    initial._init_random_number_generator(seeds=np.arange(5101, 5117, dtype=np.uint32))
    states = [[int(getattr(initial, f"_rng_s{i}")[j]) for i in range(1, 5)] for j in range(16)]
    uniforms = np.array([first_uniform(state) for state in states])
    assert np.all(uniforms > 0)
    compare("first uniforms", xt.RandomUniform(_context=CTX).generate(
            n_samples=16, particles=initial.copy()).ravel(), uniforms,
            "independent 32-bit Tausworthe/LCG recurrence", control=True, rtol=0)
    exponentials = np.array([-math.log(u) for u in uniforms])
    # Preserve Everest's proton cross-section model, including its historical
    # proton mass and initial E≈pc proxy. Only density normalization is at issue.
    energy_gev = math.hypot(450e9, float(initial.mass0)) / 1e9
    log_s = math.log(2 * 0.938271998 * energy_gev)
    pp_total = 0.041084 - 0.0023302 * log_s + 0.00031514 * log_s**2
    outer = np.array([2 * math.pi * 0.415 * (3 / (4 * math.pi))**(1 / 3) * a**(1 / 3)
                      for a in (28.085, 12.011)])
    sigma_each = np.array([0.664, 0.337]) + outer * (pp_total - 0.04) + [0.000390, 0.000076]
    densities = number_densities(3.21, [28.085, 12.011], [1, 1])
    rate = float(np.dot(densities, sigma_each) * 1e-28)
    interaction_length = 1 / rate
    flights = exponentials * interaction_length
    correct = np.minimum(flights, 1.0)
    print(f"ORACLE n_i/m3={densities} sigma_i/barn={sigma_each} "
          f"rate/m={rate:.17g} lambda/m={interaction_length:.17g}")
    for name, composition in forms:
        mat = xc.Material(components=[si, carbon], name="SiC_" + name, **params, **composition)
        assert mat.full_everest_supported, name
        np.testing.assert_allclose(mat.atomic_fractions, [0.5, 0.5], rtol=RTOL, atol=0)
        block = xc.EverestBlock(_context=CTX, length=1.0, material=mat, name=name)
        record = xc.InteractionRecord(elements=[block], names=[name], capacity=2000,
                                     record_impacts=False, record_exits=False, record_scatterings=True)
        particles = initial.copy()
        block.track(particles)
        frame = record.to_pandas()
        record.stop(elements=[block])
        assert len(frame) < record.capacity
        first = frame[frame.interaction_type == "Multiple Coulomb Scattering"].groupby(
            "particle_id_before", sort=True).first()
        assert list(first.index) == list(range(16))
        assert np.all(first.s_before.to_numpy() == 0.0)
        # Each first MCS interval ends at the sampled nuclear collision or exit.
        observed = first.s_after.to_numpy()
        control = name in ("counts_half_half", "mass_fractions")
        compare(name + " first nuclear flights/m", observed, correct,
                "min(-log(U)/sum(n_i*sigma_i*1e-28),1 m)",
                control=control, rtol=8e-15, atol=2e-16)
        print(f"{name} first particle: observed_lambda/m={observed[0] / exponentials[0]:.17g} "
              f"expected_lambda/m={interaction_length:.17g} observed_flight/m={observed[0]:.17g} "
              f"expected_flight/m={correct[0]:.17g} U={uniforms[0]:.17g}")
        print(f"{name} first collisions before exit: observed={np.sum(observed < 1)} "
              f"expected={np.sum(flights < 1)} derivation=sampled flight below 1 m")
    kernels()
    print(f"COMPARISONS total={len(matches)} pass={sum(matches)} fail={len(matches) - sum(matches)}")
    print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")


if __name__ == "__main__":
    main()
