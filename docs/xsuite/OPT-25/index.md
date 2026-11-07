---
title: "OPT-25: Periodic Twiss converts canonical time slip with reference-orbit factors"
hide:
  - toc
---

# OPT-25 — Periodic Twiss converts canonical time slip with reference-orbit factors

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Source | [_add_periodic_solution_data_to_twiss_result, twiss_postprocess_and_complem_results.py:31–87](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss_postprocess_and_complem_results.py#L31-L87)<br>[transfer_matrices.py:54–64](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/transfer_matrices.py#L54-L64)<br>[96–108](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/transfer_matrices.py#L96-L108)<br>[samples ordinary off-momentum Twiss and copies compaction, lines 12–30](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/non_linear_chromaticity.py#L12-L30)<br>[49–59](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/non_linear_chromaticity.py#L49-L59)<br>[twiss.py:144–147](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L144-L147)<br>[328–331](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L328-L331)<br>[periodic guard, 618–622](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L618-L622)<br>[off-momentum guide, twiss.rst:58–66](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/twiss.rst#L58-L66)<br>[exact longitudinal coordinates, xtrack.tex:334–381](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L334-L381)<br>[local optics vector, 4285–4293](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L4285-L4293)<br>[δ ≃ pζ approximation, 383–389](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L383-L389) |

Periodic Twiss reports incorrect slip factor η and momentum compaction αp when the closed orbit's momentum or turn time differs from the reference. It treats the one-turn advance Z of ζ = s − β₀ct (β₀ the reference speed over c) per unit canonical energy deviation pζ as a derivative in relative momentum δ, divides by the design length L instead of L − Z, and uses the reference Lorentz factor in the speed term. For 1 GeV/c protons at δ = 0.01, a constant-length cell reports αp = 0.01377 instead of zero and a dispersive ring 0.5352 instead of 0.5260, 1.762% high, by analytic flight time and tracked path and time derivatives. Off-momentum 6D RF fixed points and nonlinear-chromaticity samples share the error; orbits at reference momentum and design turn time are correct.

Run `python OPT-25.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-25.py"

    [Download OPT-25.py](OPT-25.py)

    ```python
    """OPT-25: Periodic Twiss converts canonical time slip with reference-orbit factors

    Periodic Twiss reports incorrect slip factor η and momentum compaction αp when the closed orbit's momentum or turn time differs from the reference. It treats the one-turn advance Z of ζ = s − β₀ct (β₀ the reference speed over c) per unit canonical energy deviation pζ as a derivative in relative momentum δ, divides by the design length L instead of L − Z, and uses the reference Lorentz factor in the speed term. For 1 GeV/c protons at δ = 0.01, a constant-length cell reports αp = 0.01377 instead of zero and a dispersive ring 0.5352 instead of 0.5260, 1.762% high, by analytic flight time and tracked path and time derivatives. Off-momentum 6D RF fixed points and nonlinear-chromaticity samples share the error; orbits at reference momentum and design turn time are correct.

    The script compares a straight proton cell with exact mass-shell time slip and a dispersive ring with fourth-order differences of tracked length and time. It checks default nonlinear-chromaticity samples, a nominal-momentum orbit with a changed path length, an ultrarelativistic ring and a stable displaced 6D RF fixed point. Centered reference orbits, direct tracking, the canonical matrix and independently tracked transverse tunes supply controls.
    """

    from _prebuilt import CTX, identity, kernels, np, report, xt
    from scipy.constants import c

    ATOL = 1e-9


    def straight():
        line = xt.Line(elements=[
            xt.Drift(length=0.6),
            xt.Quadrupole(length=0.3, k1=0.7),
            xt.Drift(length=1.2),
            xt.Quadrupole(length=0.3, k1=-0.7),
            xt.Drift(length=0.6),
        ])
        line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)
        line.build_tracker(_context=CTX)
        return line


    def ring(*, scale=1.0, p0c=1e9):
        elements = []
        for _ in range(8):
            elements.extend([
                xt.Multipole(knl=[0, 0.25]),
                xt.Drift(length=0.4),
                xt.Bend(length=2, angle=np.pi / 8, k0=scale * np.pi / 16,
                        model="bend-kick-bend"),
                xt.Drift(length=0.4),
                xt.Multipole(knl=[0, -0.3]),
                xt.Drift(length=0.4),
                xt.Bend(length=2, angle=np.pi / 8, k0=scale * np.pi / 16,
                        model="bend-kick-bend"),
                xt.Drift(length=0.4),
            ])
        line = xt.Line(elements=elements)
        line.particle_ref = xt.Particles(p0c=p0c, mass0=xt.PROTON_MASS_EV)
        line.build_tracker(_context=CTX)
        return line


    def orbit_measures(line, delta):
        particle = line.find_closed_orbit(delta0=delta)
        initial = np.array([particle.x[0], particle.px[0], particle.y[0], particle.py[0]])
        zeta0 = float(particle.zeta[0])
        line.track(particle)
        final = np.array([particle.x[0], particle.px[0], particle.y[0], particle.py[0]])
        np.testing.assert_allclose(final, initial, rtol=0, atol=1e-10)
        assert abs(float(particle.delta[0]) - delta) < 1e-13
        p0c = float(particle.p0c[0])
        beta = p0c * (1 + delta) / np.hypot(p0c * (1 + delta), particle.mass0)
        zeta = float(particle.zeta[0]) - zeta0
        # ζ = L − β₀cT and C = βcT for a constant-energy closed orbit.
        time = (line.get_length() - zeta) / (float(particle.beta0[0]) * c)
        return np.array([beta * c * time, time, zeta])


    def derivatives(line, delta, step):
        minus2, minus1, center, plus1, plus2 = [
            orbit_measures(line, delta + k * step) for k in (-2, -1, 0, 1, 2)
        ]
        derivative = (minus2 - 8 * minus1 + 8 * plus1 - plus2) / (12 * step)
        # Normalize each physical derivative by its own measured closed-orbit value.
        expected = np.array([derivative[0] / center[0], derivative[1] / center[1],
                             derivative[2]])
        return center, expected


    def ring_expectation(line, delta):
        _, coarse = derivatives(line, delta, 1e-3)
        center, fine = derivatives(line, delta, 5e-4)
        np.testing.assert_allclose(coarse[:2], fine[:2], rtol=0, atol=1e-10)
        np.testing.assert_allclose(coarse[2], fine[2], rtol=0, atol=2e-8)
        return center, fine


    def static_canonical_response(twiss):
        matrix = twiss.R_matrix
        # Transverse periodicity at fixed pζ gives (I−R₄)D = R[:4,5].
        dispersion = np.linalg.solve(np.eye(4) - matrix[:4, :4], matrix[:4, 5])
        return float(matrix[4, 5] + matrix[4, :4] @ dispersion)


    def physical_conversion(line, twiss, center, expected):
        p0c = float(line.particle_ref.p0c[0])
        mass = float(line.particle_ref.mass0)
        u = 1 + float(twiss.particle_on_co.delta[0])
        # dpζ/dδ = β/β₀, η = −Z′/(L−Z), αp = η + d(log β)/dδ.
        zprime = float(twiss.particle_on_co.rvv[0]) * static_canonical_response(twiss)
        eta = -zprime / (line.get_length() - center[2])
        alpha = eta + mass**2 / (u * (mass**2 + (p0c * u)**2))
        np.testing.assert_allclose([alpha, eta], expected[:2], rtol=0, atol=ATOL)
        np.testing.assert_allclose(zprime, expected[2], rtol=0, atol=2e-8)


    def observed(twiss):
        return np.array([twiss.momentum_compaction_factor, twiss.slip_factor,
                         twiss.slip_factor_dzeta_ddelta], dtype=float)


    def tracked_tunes(line, delta):
        step = 1e-7
        coordinates = np.concatenate((np.eye(4), -np.eye(4)), axis=1) * step
        particle = xt.Particles(
            _context=CTX, p0c=line.particle_ref.p0c[0], mass0=line.particle_ref.mass0,
            x=coordinates[0], px=coordinates[1], y=coordinates[2], py=coordinates[3],
            delta=delta,
        )
        line.track(particle)
        values = np.array([particle.x, particle.px, particle.y, particle.py])
        matrix = (values[:, :4] - values[:, 4:]) / (2 * step)
        return [np.arccos(np.trace(matrix[i:i+2, i:i+2]) / 2) / (2 * np.pi)
                for i in (0, 2)]


    def main():
        identity()
        cases = {}
        line = straight()
        length = line.get_length()
        p0c = float(line.particle_ref.p0c[0])
        mass = float(line.particle_ref.mass0)
        beta0 = float(line.particle_ref.beta0[0])
        print("scalar order=[alpha_p, eta, dZ/ddelta]; lengths in m; fixed-reference delta")
        for delta in (0.0, 0.001, 0.01, 0.2, -0.01):
            u = 1 + delta
            beta = p0c * u / np.hypot(p0c * u, mass)
            # C=L exactly; logarithmic differentiation of T=L/(βc) gives η.
            eta = -mass**2 / (u * (mass**2 + (p0c * u)**2))
            zeta = length * (1 - beta0 / beta)
            expected = np.array([0, eta, -length * beta0 / beta * eta])
            twiss = line.twiss(method="4d", delta0=delta, chrom=False)
            particle = xt.Particles(_context=CTX, p0c=p0c, mass0=mass, delta=delta)
            line.track(particle)
            tracked_zeta = float(np.asarray(particle.zeta)[0])
            np.testing.assert_allclose(tracked_zeta, zeta, rtol=0, atol=5e-15)
            np.testing.assert_allclose(beta / beta0 * (length - tracked_zeta),
                                       length, rtol=0, atol=5e-15)
            np.testing.assert_allclose(twiss.R_matrix[4, 5], expected[2] * beta0 / beta,
                                       rtol=0, atol=5e-10)
            physical_conversion(line, twiss, np.array([length, 0, zeta]), expected)
            if delta == 0:
                np.testing.assert_allclose(observed(twiss), expected, rtol=0, atol=ATOL)
            cases[f"straight delta={delta:g}; derivation=C=L,T=L/(beta*c)"] = (
                observed(twiss), expected)

        samples = line.get_non_linear_chromaticity()
        for delta, twiss in zip(samples.delta0, samples.twiss, strict=True):
            np.testing.assert_allclose([twiss.qx, twiss.qy], tracked_tunes(line, float(delta)),
                                       rtol=0, atol=1e-11)
        cases["nonlinear-chromaticity default samples; derivation=C=L at every delta"] = (
            samples.momentum_compaction_factor, np.zeros(len(samples.delta0)))
        print(f"nonlinear-chromaticity delta={samples.delta0}; transverse tune controls pass")
        for p0c, scale in ((1e9, 1.0), (1e9, 1.003), (1e12, 1.0)):
            line = ring(scale=scale, p0c=p0c)
            offsets = (0.0, 0.001, 0.01) if scale == 1 else (0.0,)
            for delta in offsets:
                twiss = line.twiss(method="4d", delta0=delta, chrom=False)
                center, expected = ring_expectation(line, delta)
                physical_conversion(line, twiss, center, expected)
                if delta == 0 and scale == 1:
                    np.testing.assert_allclose(observed(twiss)[:2], expected[:2],
                                               rtol=0, atol=ATOL)
                label = (f"ring p0c={p0c:g} scale={scale:g} delta={delta:g}; "
                         "derivation=tracked C'/C,T'/T,Z' (fourth-order)")
                cases[label] = (observed(twiss), expected)
                print(f"orbit p0c={p0c:g} scale={scale:g} delta={delta:g}: "
                      f"C={center[0]:.15g} L={line.get_length():.15g}")

        no_rf = ring(scale=1.0005)
        cavity = xt.Cavity(voltage=2e4, phase=np.pi,
                           frequency=float(no_rf.particle_ref.beta0[0]) * c / no_rf.get_length())
        rf = xt.Line(elements=[cavity] + [element.copy() for element in no_rf.elements])
        rf.particle_ref = no_rf.particle_ref.copy()
        rf.build_tracker(_context=CTX)
        twiss = rf.twiss(method="6d", chrom=False)
        delta = float(twiss.particle_on_co.delta[0])
        particle = twiss.particle_on_co.copy()
        rf.track(particle)
        assert abs(float(particle.zeta[0] - twiss.particle_on_co.zeta[0])) < 1e-10
        assert abs(float(particle.delta[0]) - delta) < 1e-12
        _, expected = ring_expectation(no_rf, delta)
        print(f"6d RF fixed point delta={delta:.16g} qs={twiss.qs:.16g}; "
              "zero net cavity gain and one-turn closure controls pass")
        cases["6d RF; derivation=tracked no-RF path/time at actual fixed-point delta"] = (
            observed(twiss), expected)
        print("direct tracking, canonical matrix, physical conversion and centered controls pass")
        kernels()
        report(cases, rtol=0, atol=2e-8)


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `claude-opus-5-5` | high | 2026-11-07 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

