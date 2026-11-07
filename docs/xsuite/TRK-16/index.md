---
title: "TRK-16: Quantum photon rate omits reference charge squared"
hide:
  - toc
---

# TRK-16 — Quantum photon rate omits reference charge squared

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [`synrad_average_number_of_photons`, synrad_spectrum.h:209–219](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/headers/synrad_spectrum.h#L209-L219)<br>[total-energy sampler](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/headers/synrad_spectrum.h#L409-L433)<br>[individual-photon sampler](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/headers/synrad_spectrum.h#L463-L499)<br>[radiation laws](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L5872-L5917) |

The `quantum` and `quantum-kick` synchrotron-radiation modes omit the square of the reference charge from the photon emission rate, underestimating emission for multiply charged reference particles. The Liénard radiated power and an independent integration of the photon spectrum require the rate to scale with charge squared at fixed curvature. For a reference particle of charge 6e (six elementary charges), rest energy 11.18 GeV and momentum 250 TeV/c over 600 m at curvature 2 × 10⁻⁵ m⁻¹, a `Multipole` emits on average 2.822 photons per particle instead of 101.8. An unsliced `Bend` with the same parameters loses 57.02 eV on average in `quantum` mode instead of 2075 eV, which mean-radiation tracking gives. Unit-charge references are correct.

Run `python TRK-16.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-16.py"

    [Download TRK-16.py](TRK-16.py)

    ```python
    """TRK-16: Quantum photon rate omits reference charge squared

    The `quantum` and `quantum-kick` synchrotron-radiation modes omit the square of the reference charge
    from the photon emission rate, underestimating emission for multiply charged reference particles.
    The Liénard radiated power and an independent integration of the photon spectrum require the rate to
    scale with charge squared at fixed curvature. For a reference particle of charge 6e (six elementary
    charges), rest energy 11.18 GeV and momentum 250 TeV/c over 600 m at curvature 2 × 10⁻⁵ m⁻¹, a
    `Multipole` emits on average 2.822 photons per particle instead of 101.8. An unsliced `Bend` with
    the same parameters loses 57.02 eV on average in `quantum` mode instead of 2075 eV, which
    mean-radiation tracking gives. Unit-charge references are correct.

    The script tracks 24,576 seeded particles through 600 m for reference charges 1, 2 and 6, records
    every photon emitted in a `Multipole` at fixed curvature and at a fixed 0.75 T field, and compares
    photon counts and energies with the Liénard power divided by the mean photon energy from
    Bessel-function quadrature of the spectrum. It then compares the mean energy loss in an unsliced
    `Bend` in the `mean`, `quantum` and `quantum-kick` modes with the same power. Unit-charge counts and
    spectra must agree within six standard errors, and so must the mean mode for all charges. It runs
    serially on the installed prebuilt CPU kernels with kernel compilation forbidden.
    """

    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.constants import c, e, epsilon_0, hbar
    from scipy.integrate import quad
    from scipy.special import kv


    def line(*elements):
        result = xt.Line(elements=list(elements))
        result.build_tracker(_context=CTX)
        return result


    identity(radiation="mean/quantum/quantum-kick", random_seed="5237+997*arange(24576)")
    print("ORACLE SI Lienard power: Q^2*c*beta^4*gamma^4*curvature^2/(6*pi*epsilon0)")
    print("Expected photons = power*length/(beta*c*mean_photon_energy); spectrum via Bessel quadrature")
    n = 24576
    mass = 11177929613.0  # A carbon-12 nucleus, with several reference charge states.
    p0c = 2.5e14
    length = 600.0
    curvature = 2e-5
    seeds = (5237 + 997 * np.arange(n)).astype(np.uint32)
    # F(u)=integral_u^infinity K_(5/3)(t) dt. Interchange the integrals:
    # integral u^j F(u) du = integral t^(j+1) K_(5/3)(t) dt / (j+1).
    # t=u^3 removes the integrable t^(-2/3) endpoint singularity in I_0.
    moments = np.array(
        [
            quad(
                lambda u, j=j: 3 * u ** (3 * j + 5) * kv(5 / 3, u**3) / (j + 1),
                0,
                np.inf,
                epsabs=1e-12,
                epsrel=1e-12,
            )[0]
            for j in range(3)
        ]
    )
    mean_u, second_u = moments[1:] / moments[0]
    np.testing.assert_allclose([mean_u, second_u], [8 * np.sqrt(3) / 45, 11 / 27], rtol=2e-11)
    print("SPECTRUM_QUADRATURE", moments.tolist(), "mean_u", mean_u, "second_u", second_u)


    def oracle(charge, curv):
        gamma = np.hypot(p0c, mass) / mass
        beta = p0c / np.hypot(p0c, mass)
        # Lienard radiation power for perpendicular circular acceleration. The photon count
        # N = P L/(beta c <E_photon>) scales as q0^2 at fixed curvature, q0^3 at fixed field.
        power = (charge * e) ** 2 * c * beta**4 * gamma**4 * curv**2 / (6 * np.pi * epsilon_0)
        loss = power * length / (beta * c) / e
        critical = 1.5 * hbar * c / e * gamma**3 * beta * abs(curv)
        photon_mean = critical * mean_u
        rate = loss / photon_mean
        return dict(
            gamma=gamma,
            beta=beta,
            B_T=curv * p0c / (charge * c),
            loss_eV=loss,
            critical_eV=critical,
            mean_photon_eV=photon_mean,
            photons=rate,
            second_photon_eV2=critical**2 * second_u,
        )


    def particles(charge):
        p = xt.Particles(_context=CTX, p0c=p0c, mass0=mass, q0=charge, x=np.zeros(n))
        assert np.all(p.charge_ratio == 1) and np.all(p.mass_ratio == 1)
        assert np.all(p.chi == 1)
        p._init_random_number_generator(seeds=seeds)
        return p


    def count_case(charge, curv, label):
        expected = oracle(charge, curv)
        angle = curv * length
        lattice = line(xt.Multipole(knl=[angle], hxl=angle, length=length))
        lattice.configure_radiation(model="quantum")
        capacity = 1_000_000
        record = lattice.start_internal_logging_for_elements_of_type(xt.Multipole, capacity=capacity)
        p = particles(charge)
        lattice.track(p)
        assert np.all(p.state == 1)
        count = int(record._index.num_recorded)
        assert 0 < count < capacity
        photon_energy = np.asarray(record.photon_energy[:count]).copy()
        ids = np.asarray(record.particle_id[:count]).astype(int)
        per_particle = np.bincount(ids, minlength=n)
        assert len(per_particle) == n
        rate = count / n
        sem = np.sqrt(count) / n
        spectral_sem = np.std(photon_energy, ddof=1) / np.sqrt(count)
        assert abs(photon_energy.mean() - expected["mean_photon_eV"]) < 6 * spectral_sem
        assert abs(rate - expected["photons"] / charge**2) < 6 * sem
        zeros = float(np.mean(per_particle == 0))
        zero_sem = np.sqrt(zeros * (1 - zeros) / n)
        model_zeros = np.exp(-expected["photons"] / charge**2)
        assert abs(zeros - model_zeros) < 6 * zero_sem
        if abs(charge) != 1:
            assert abs(rate - expected["photons"]) > 100 * sem
        print(
            f"PHOTONS {label} q0={charge} count={count} rate={rate:.12g} "
            f"expected={expected['photons']:.12g} sem={sem:.12g}"
        )
        print(
            f"SPECTRUM q0={charge} mean_eV={photon_energy.mean():.12g} "
            f"expected_eV={expected['mean_photon_eV']:.12g} sem_eV={spectral_sem:.12g} "
            f"zero_fraction={zeros:.12g} expected_zero={np.exp(-expected['photons']):.12g}"
        )
        lattice.stop_internal_logging_for_elements_of_type(xt.Multipole)
        return per_particle, photon_energy


    previous_counts = None
    previous_energies = None
    for charge in (1, 2, 6):
        counts, energies = count_case(charge, curvature, "fixed curvature")
        if previous_counts is not None:
            assert previous_energies is not None
            np.testing.assert_array_equal(counts, previous_counts)
            np.testing.assert_allclose(energies, previous_energies, rtol=3e-15, atol=0)
        previous_counts, previous_energies = counts, energies
    print("FIXED_CURVATURE_IDENTICAL_PHOTON_COUNTS_AND_ENERGIES", [1, 2, 6])

    for charge in (1, 2, 6):
        curv = 0.75 * charge * c / p0c
        count_case(charge, curv, "fixed B=0.75T")

    for charge in (1, 2, 6):
        expected = oracle(charge, curvature)
        lattice = line(
            xt.Bend(
                length=length,
                angle=curvature * length,
                k0=curvature,
                model="bend-kick-bend",
                integrator="uniform",
                num_multipole_kicks=1,
                edge_entry_active=False,
                edge_exit_active=False,
            )
        )
        for mode in ("mean", "quantum", "quantum-kick"):
            p = particles(charge)
            before_ptau = np.asarray(p.ptau).copy()
            lattice.configure_radiation(model=mode)
            lattice.track(p)
            assert np.all(p.state == 1)
            losses = -(np.asarray(p.ptau) - before_ptau) * p0c
            measured = float(losses.mean())
            sem = float(losses.std(ddof=1) / np.sqrt(n))
            target = expected["loss_eV"] if mode == "mean" else expected["loss_eV"] / charge**2
            # Updating normalized momentum quantizes these tiny relative losses.
            rounding_bound = 3 * np.finfo(float).eps * p0c
            assert abs(measured - target) < 6 * sem + rounding_bound
            if charge != 1 and mode != "mean":
                assert abs(measured - expected["loss_eV"]) > 100 * sem
            print(
                f"BEND q0={charge} mode={mode} loss_eV={measured:.12g} "
                f"expected_eV={expected['loss_eV']:.12g} sem_eV={sem:.12g} "
                f"ratio={measured / expected['loss_eV']:.12g}"
            )

    print("CONTROLS unit-charge photon rates and spectra within 6 sampling standard errors")
    print("ENERGY_ROUNDING_BOUND_eV", 3 * np.finfo(float).eps * p0c)
    kernels()
    print("DEFECT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-28 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, serial |

