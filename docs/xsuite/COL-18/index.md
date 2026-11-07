---
title: "COL-18: Round-beam variance derivatives use an absolute near-axis cutoff"
hide:
  - toc
---

# COL-18 — Round-beam variance derivatives use an absolute near-axis cutoff

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-10-29 on xfields main (`3f06c87`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`compute_Gx_Gy`, compute_gx_gy.h:25–35](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/fieldmaps/bigaussian_src/compute_gx_gy.h#L25-L35)<br>[synchro-beam energy kick](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/beam_elements/beambeam_src/beambeam3d.h#L263-L281)<br>[Gaussian potential derivatives and round expressions](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L938-L994)<br>[longitudinal map](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L1005-L1038) |

For round beam slices whose width changes along the collision, `compute_Gx_Gy` uses on-axis variance derivatives inside a fixed radius of 100 nm, corrupting the width contribution of 3D beam-beam to `pzeta`. Gauss's law and the Gaussian heat equation require the radial dependence exp(−r²/(2σ²)), with σ the RMS beam size; integrating the longitudinal derivative of the executed transverse kick along the radius confirms it. For 9 GeV/c positrons and an expanding 35 nm slice, the width kick at r = 2σ is 7.840 × 10⁻¹¹ instead of 1.061 × 10⁻¹¹, a factor e² too large; at σ = 35 µm and r = 99.9 nm the relative error is 4.073 × 10⁻⁶. Transverse kicks are correct.

Run `python COL-18.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-18.py"

    [Download COL-18.py](COL-18.py)

    ```python
    """COL-18: Round-beam variance derivatives use an absolute near-axis cutoff

    For round beam slices whose width changes along the collision, `compute_Gx_Gy` uses on-axis variance
    derivatives inside a fixed radius of 100 nm, corrupting the width contribution of 3D beam-beam to
    `pzeta`. Gauss's law and the Gaussian heat equation require the radial dependence exp(−r²/(2σ²)),
    with σ the RMS beam size; integrating the longitudinal derivative of the executed transverse kick
    along the radius confirms it. For 9 GeV/c positrons and an expanding 35 nm slice, the width kick at
    r = 2σ is 7.840 × 10⁻¹¹ instead of 1.061 × 10⁻¹¹, a factor e² too large; at σ = 35 µm and r = 99.9
    nm the relative error is 4.073 × 10⁻⁶. Transverse kicks are correct.

    The script tracks positrons through a weak-strong `BeamBeamBiGaussian3D` slice with covariance
    blocks v·[[1, a], [a, a² + 40000]], v = σ², a = 700 m⁻¹, at radii including zero, 2σ and both sides
    of the cutoff, and compares the width kick with the Gaussian prediction; controls vary width, charge
    sign, expansion, intensity and momentum. As an independent check it differentiates the executed
    radial kick with respect to ζ at signed offsets, integrates it from 0 to 2σ with 48 Gauss–Legendre
    points and compares with the analytic radial difference.
    """

    # The execution helper must set thread limits before numerical imports.
    # isort: off
    from _prebuilt import CTX, identity, kernels, np, xt
    import xfields as xf

    # isort: on
    from numpy.polynomial.legendre import leggauss

    # Match the pinned electromagnetic constants used by the executed kernel.
    EPS = 8.854187817620e-12
    QE = 1.60217662e-19


    def track(sigma, a, radii, *, zeta=0.0, population=7e5, p0c=9e9, q=1, threshold=1e-28):
        variance = sigma**2
        element = xf.BeamBeamBiGaussian3D(
            _context=CTX,
            flag_beamstrahlung=0,
            flag_bhabha=0,
            phi=0,
            alpha=0,
            other_beam_q0=q,
            slices_other_beam_num_particles=[population],
            slices_other_beam_zeta_center=[0],
            slices_other_beam_Sigma_11=[variance],
            slices_other_beam_Sigma_12=[a * variance],
            slices_other_beam_Sigma_22=[(a * a + 4e4) * variance],
            slices_other_beam_Sigma_33=[variance],
            slices_other_beam_Sigma_34=[a * variance],
            slices_other_beam_Sigma_44=[(a * a + 4e4) * variance],
            threshold_singular=threshold,
        )
        angles = np.arange(len(radii)) * 0.47
        particles = xt.Particles(
            _context=CTX,
            p0c=p0c,
            mass0=xt.ELECTRON_MASS_EV,
            x=radii * np.cos(angles),
            y=radii * np.sin(angles),
            zeta=zeta,
        )
        element.track(particles)
        assert np.all(particles.state == 1)
        radial_kick = particles.px * np.cos(angles) + particles.py * np.sin(angles)
        width_kick = particles.pzeta - (particles.px**2 + particles.py**2) / 4
        return radial_kick, width_kick


    def main():
        identity()
        print("ORACLE Fz=K*a*exp(-r^2/(2*sigma^2))/(4*pi*epsilon0); K=N*q_s*q0/(P0*c)")
        print("Width kick removes (Fx^2+Fy^2)/4; v_prime(S=0)=2*a*v; dS/dzeta=1/2")
        records = []
        # Radii straddle the fixed 100 nm branch; normalized radii also test geometric similarity.
        for sigma, a, q, population, p0c, threshold in [
            (35e-9, 700.0, 1, 7e5, 9e9, 1e-28),
            (35e-6, 700.0, 1, 7e5, 9e9, 1e-28),
            (70e-9, -350.0, -1, 1.4e6, 18e9, 1e-40),
            (35e-9, -700.0, 1, 7e5, 9e9, 1e-28),
            (35e-9, 700.0, -1, 7e5, 9e9, 1e-28),
            (35e-9, 700.0, 1, 2.1e6, 4.5e9, 1e-28),
            (35e-9, 0.0, 1, 7e5, 9e9, 1e-28),
        ]:
            radii = np.array([0, 0.5 * sigma, sigma, 2 * sigma, 3 * sigma, 99.9e-9, 100.1e-9])
            radial, width = track(
                sigma, a, radii, q=q, population=population, p0c=p0c, threshold=threshold
            )
            # Gauss's law and the Gaussian heat equation (dv/dS = 2*a*v, dS/dzeta = 1/2) give the width
            # kick K*a*exp(-r^2/(2*sigma^2))/(4*pi*epsilon0), K = N*q_s*q0/(P0*c) in SI units.
            factor = population * QE * q / p0c / (4 * np.pi * EPS)
            oracle = factor * a * np.exp(-(radii**2) / (2 * sigma**2))
            radial_oracle = np.zeros_like(radii)
            radial_oracle[1:] = 2 * factor * -np.expm1(-(radii[1:] ** 2) / (2 * sigma**2)) / radii[1:]
            np.testing.assert_allclose(radial, radial_oracle, rtol=4e-10, atol=1e-22)
            rec = dict(
                sigma=sigma,
                a=a,
                q=q,
                population=population,
                p0c=p0c,
                threshold=threshold,
                radii=radii.tolist(),
                measured=width.tolist(),
                oracle=oracle.tolist(),
                transverse_max_abs_error=float(np.max(abs(radial - radial_oracle))),
            )
            records.append(rec)
            print("WIDTH", rec)
            print(
                f"TWO_SIGMA sigma={sigma:.9g} a={a:g} q={q} actual={width[3]:.17g} expected={oracle[3]:.17g}"
            )
            if sigma == 35e-9 and a == 700 and q == 1 and population == 7e5:
                np.testing.assert_allclose(width[3] / oracle[3], np.exp(2), rtol=3e-14)
                np.testing.assert_allclose(width[:4], width[0], rtol=3e-14, atol=1e-25)
            if a == 0:
                np.testing.assert_allclose(width, 0, atol=1e-25)
            elif sigma > 1e-6:
                resolved = (radii >= 1e-7) | (radii == 0)
                np.testing.assert_allclose(width[resolved], oracle[resolved], rtol=1e-10, atol=1e-24)
            else:
                np.testing.assert_allclose(width[-1], oracle[-1], rtol=1e-12, atol=1e-24)
                assert abs(width[-2] - oracle[-2]) > 0.1 * abs(width[-2])

        # A potential's mixed derivatives commute. Integrate d(F_r)/d(zeta) from 0 to 2 sigma.
        # This oracle uses *executed transverse kicks* at two zeta offsets, with no G implementation.
        nodes, weights = leggauss(48)
        mixed = []
        for sigma in [35e-9, 35e-6]:
            a = 700.0
            radii = sigma * (nodes + 1)
            observed = track(sigma, a, np.array([0.0, 2 * sigma]))[1]
            expected_delta = 7e5 * QE / 9e9 / (4 * np.pi * EPS) * a * np.expm1(-2)
            for step in [1e-7, 5e-8]:
                plus = track(sigma, a, radii, zeta=step)[0]
                minus = track(sigma, a, radii, zeta=-step)[0]
                fd_integral = float(sigma * np.dot(weights, (plus - minus) / (2 * step)))
                np.testing.assert_allclose(fd_integral, expected_delta, rtol=5e-9)
                rec = dict(
                    sigma=sigma,
                    zeta_step=step,
                    mixed_integral=fd_integral,
                    analytic_delta=expected_delta,
                    measured_delta=float(observed[1] - observed[0]),
                )
                mixed.append(rec)
                print("MIXED_DERIVATIVE", rec)
        for index in (0, 2):
            assert abs(mixed[index + 1]["mixed_integral"] - mixed[index + 1]["analytic_delta"]) < abs(
                mixed[index]["mixed_integral"] - mixed[index]["analytic_delta"]
            )
        assert abs(mixed[1]["measured_delta"] - mixed[1]["analytic_delta"]) > 6e-11
        np.testing.assert_allclose(mixed[3]["measured_delta"], mixed[3]["analytic_delta"], rtol=1e-12)
        kernels()
        print("DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-29 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-29 | xfields 0.27.5 (`223b4fa134a1`), installed wheel, prebuilt kernels, compilation forbidden, CPU, serial |

