---
title: "COL-27: Equal positive x–z collision eigenvalues make Nagaitsev IBS rates NaN"
hide:
  - toc
---

# COL-27 — Equal positive x–z collision eigenvalues make Nagaitsev IBS rates NaN

| | |
|---|---|
| Code | [Xsuite](../index.md) · Collective effects |
| Priority | P3 |
| Severity | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xfields main (`3f06c87`, head build) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`NagaitsevIBS.integrals`, _analytical.py:588–635](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_analytical.py#L588-L635)<br>[integrals contract, lines 523–575](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_analytical.py#L523-L575)<br>[API signature, _api.py:30–42](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_api.py#L30-L42)<br>[dispatch, lines 88–110](https://github.com/xsuite/xfields/blob/223b4fa134a1435be9b45ef21ae8903f96c5f685/xfields/ibs/_api.py#L88-L110)<br>[convergent positive Carlson integral and its full-equality identity,
xfields.tex:2520–2544](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L2520-L2544)<br>[amplitude convention, lines 2493–2504](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L2493-L2504)<br>[emittance-rate equations, lines 2607–2625](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xfields.tex#L2607-L2625) |

Requesting Nagaitsev intrabeam-scattering rates with equal positive horizontal and longitudinal collision eigenvalues returns NaN horizontal and longitudinal rates. A dispersion-free proton ring at reference Lorentz factor γ₀=2.000 should give amplitude growth rates (−7.758×10⁻⁵, 6.207×10⁻⁴, −7.758×10⁻⁵) s⁻¹, but only the vertical rate remains finite. Direct integration of the positive collision matrix establishes the finite result. A regular quadrupole-and-drift lattice also fails when equality occurs at one sampled location. Adjacent representable nonzero eigenvalue gaps remain accurate.

Run `python COL-27.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: COL-27.py"

    [Download COL-27.py](COL-27.py)

    ```python
    """COL-27: Equal positive x–z collision eigenvalues make Nagaitsev IBS rates NaN

    Requesting Nagaitsev intrabeam-scattering rates with equal positive horizontal and longitudinal collision eigenvalues returns NaN horizontal and longitudinal rates. A dispersion-free proton ring at reference Lorentz factor γ₀=2.000 should give amplitude growth rates (−7.758×10⁻⁵, 6.207×10⁻⁴, −7.758×10⁻⁵) s⁻¹, but only the vertical rate remains finite. Direct integration of the positive collision matrix establishes the finite result. A regular quadrupole-and-drift lattice also fails when equality occurs at one sampled location. Adjacent representable nonzero eigenvalue gaps remain accurate.

    The script builds dispersion-free periodic optics with a segment map and a quadrupole-and-drift lattice, then compares native integrals and public growth rates with direct 70-digit positive collision-matrix quadrature. Exact equality, full isotropy, adjacent representable emittances and vertical-only equality distinguish the missing finite limit from ordinary rounding. It uses serial prebuilt kernels with compilation forbidden and needs no random seed.
    """

    import json
    import math
    import warnings

    import mpmath as mp
    from _prebuilt import CTX, identity, kernels, np, xt
    from scipy.constants import c
    from xfields.ibs import NagaitsevIBS

    mp.mp.dps = 70
    TOLERANCE = 5e-13


    def exact(value):
        numerator, denominator = float(value).as_integer_ratio()
        return mp.mpf(numerator) / denominator


    def make_map(beta=8):
        line = xt.Line(elements=[xt.LineSegmentMap(
            length=128, qx=0.13, qy=0.21, betx=beta, bety=beta,
            longitudinal_mode="frozen",
        )])
        line.particle_ref = xt.Particles(_context=CTX, gamma0=2)
        line.build_tracker(_context=CTX)
        return line.twiss(method="4d")


    def positive_tensor_integral(lambdas):
        # Eigenvalues are ordered x,z,y. Substituting u=1/t in the positive
        # Carlson integral gives R_i=(3/2)*sqrt(product(lambda))*integral_0^inf
        # sqrt(u)/((u+lambda_i)*sqrt(product(u+lambda))) du. Scaling u by the
        # largest eigenvalue leaves the prefactor sqrt(scale). No gap is divided.
        scale = max(lambdas)
        lam = [value / scale for value in lambdas]
        prefactor = mp.mpf("1.5") * mp.sqrt(mp.fprod(lam)) * mp.sqrt(scale)
        rr = [prefactor * mp.quad(
            lambda u, i=i: mp.sqrt(u) / ((u + lam[i]) * mp.sqrt(mp.fprod(u + v for v in lam))),
            [0, 1, 4, mp.inf],
        ) for i in range(3)]
        return [sum(rr) - 3 * value for value in rr]


    def expected_integrals(tw, epsx, epsy, sigma_delta):
        rows = []
        for j in range(len(tw.s) - 1):
            assert tw.dx[j] == tw.dy[j] == tw.dpx[j] == tw.dpy[j] == 0
            ax = exact(tw.betx[j]) / exact(epsx)
            ay = exact(tw.bety[j]) / exact(epsy)
            az = exact(tw.gamma0)**2 / exact(sigma_delta)**2
            sx, sz, sy = positive_tensor_integral([ax, az, ay])
            size = mp.sqrt(exact(tw.betx[j]) * exact(epsx) * exact(tw.bety[j]) * exact(epsy))
            # Native lattice integration uses left-endpoint path weights. Only
            # the finite collision tensor is replaced in the independent oracle.
            weight = (exact(tw.s[j + 1]) - exact(tw.s[j])) / exact(tw.line_length) / size
            rows.append([exact(tw.betx[j]) * sx * weight,
                         exact(tw.bety[j]) * sy * weight,
                         exact(tw.gamma0)**2 * sz * weight])
        return np.array([float(sum(row[i] for row in rows)) for i in range(3)])


    def scaled_error(observed, expected):
        if not np.isfinite(observed).all():
            return math.inf
        return float(max(abs(observed - expected)) / max(max(abs(expected)), 1))


    def measure(label, tw, epsx, epsy, sigma_delta, *, rates=False, control=False):
        obj = NagaitsevIBS(tw)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = np.array(obj.integrals(
                gemitt_x=epsx, gemitt_y=epsy, sigma_delta=sigma_delta,
            ).as_tuple())
        want = expected_integrals(tw, epsx, epsy, sigma_delta)
        assert np.isfinite(want).all()
        # The repeated final endpoint has zero path weight; equality there can
        # emit a warning but does not contaminate the lattice integral.
        gaps = np.abs(tw.betx[:-1] / epsx - tw.gamma0**2 / sigma_delta**2)
        error = scaled_error(got, want)
        matches = error <= TOLERANCE
        record = dict(
            case=label, epsx=epsx, epsy=epsy, sigma_delta=sigma_delta,
            min_gap=float(min(gaps)), zero_gap_rows=np.where(gaps == 0)[0].tolist(),
            observed=got.tolist(), expected=want.tolist(),
            derivation="70-digit positive collision-matrix quadrature; left-endpoint path weights",
            scaled_error=error, matches=matches,
            warnings=sorted({str(w.message) for w in caught}),
        )
        if control:
            assert matches, record
        if rates:
            settings = dict(gemitt_x=epsx, gemitt_y=epsy, sigma_delta=sigma_delta,
                            bunch_length=0.35, total_beam_intensity=4e9)
            with warnings.catch_warnings(record=True) as rate_warnings:
                warnings.simplefilter("always")
                got_rates = np.array(tw.get_ibs_growth_rates(
                    formalism="Nagaitsev", **settings,
                ).as_tuple())
            # At zero dispersion the pending dispersed-cutoff normalization does
            # not enter. Retain the native Coulomb log to isolate the tensor.
            # The manual gives amplitude rates, half the emittance growth rates.
            coulomb_log = obj.coulomb_log(
                gemitt_x=epsx, gemitt_y=epsy, sigma_delta=sigma_delta,
                bunch_length=0.35, total_beam_intensity=4_000_000_000,
            )
            radius = obj._particle.get_classical_particle_radius0()
            prefactor = (4e9 * radius**2 * c * coulomb_log
                         / (24 * math.pi * tw.beta0**3 * tw.gamma0**5 * 0.35))
            want_rates = want * prefactor / np.array([epsx, epsy, sigma_delta**2])
            rate_error = scaled_error(got_rates, want_rates)
            matches &= rate_error <= TOLERANCE * max(abs(want_rates))
            record.update(observed_rates=got_rates.tolist(), expected_rates=want_rates.tolist(),
                          rate_derivation="N*r0^2*c*logLambda*I/(24*pi*beta0^3*gamma0^5*sigma_z*emittance)",
                          coulomb_log=coulomb_log, matches=bool(matches),
                          rate_warnings=sorted({str(w.message) for w in rate_warnings}))
        print(json.dumps(record), flush=True)
        return not matches, error


    def main():
        identity()
        print("mpmath", mp.__version__, "digits", mp.mp.dps)
        tw = make_map()
        sigma_delta = 2.**-10
        # Power-of-two scaling of the stored beta makes lambda_x=lambda_z=2^22
        # exactly, with positive emittances and lambda_y=2^24.
        epsx = math.ldexp(float(tw.betx[0]), -22)
        epsy = math.ldexp(float(tw.bety[0]), -24)
        print("OPTICS beta_x", tw.betx, "beta_y", tw.bety, "gamma0", tw.gamma0)
        reference = expected_integrals(tw, epsx, epsy, sigma_delta)
        with mp.workdps(90):
            finer = expected_integrals(tw, epsx, epsy, sigma_delta)
        refinement = scaled_error(reference, finer)
        print("QUADRATURE_REFINEMENT_70_TO_90_DIGITS", refinement, "expected_max", 1e-15)
        assert refinement < 1e-15
        defect, _ = measure("constant_positive_xz_equality", tw, epsx, epsy, sigma_delta, rates=True)
        failure, _ = measure("full_isotropy", tw, epsx,
                             math.ldexp(float(tw.bety[0]), -22), sigma_delta)
        defect |= failure
        control_errors = []
        for eta in (-1e-6, 1e-6, -1e-10, 1e-10, -1e-14, 1e-14):
            _, error = measure(f"near_equality_{eta}", tw, epsx * (1 + eta), epsy,
                               sigma_delta, control=True)
            control_errors.append(error)
        for direction in (0., np.inf):
            _, error = measure(f"adjacent_emittance_{direction}", tw,
                               float(np.nextafter(epsx, direction)), epsy, sigma_delta, control=True)
            control_errors.append(error)
        _, error = measure("vertical_longitudinal_equality_control", tw, epsx / 4, epsx,
                           sigma_delta, control=True)
        control_errors.append(error)
        print("FINITE_CONTROLS", len(control_errors), "max_scaled_error", max(control_errors),
              "bound", TOLERANCE)
        # A normal quadrupole/drift ring reaches exact equality at only one
        # positively weighted sampled row; constant optics are not required.
        line = xt.Line(elements=[xt.Drift(length=1), xt.Quadrupole(length=.2, k1=.7),
                                 xt.Drift(length=2), xt.Quadrupole(length=.2, k1=-.72),
                                 xt.Drift(length=1)])
        line.particle_ref = xt.Particles(_context=CTX, gamma0=2)
        line.build_tracker(_context=CTX)
        tf = line.twiss(method="4d")
        print("QUADRUPOLE_DRIFT_OPTICS s", tf.s, "beta_x", tf.betx, "beta_y", tf.bety,
              "tunes", tf.qx, tf.qy)
        failure, _ = measure("quadrupole_drift_periodic_lattice", tf,
                             math.ldexp(float(tf.betx[0]), -22),
                             math.ldexp(float(tf.bety[0]), -24), sigma_delta, rates=True)
        defect |= failure
        kernels()
        print("DEFECT REPRODUCED" if defect else "NOT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xfields 0.27.5 (`223b4fa134a1`), installed packages, host IBS calculation, prebuilt optics kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xfields 0.27.5 (`223b4fa134a1`), installed packages, host IBS calculation, prebuilt optics kernels (compilation forbidden), CPU, double, serial |

