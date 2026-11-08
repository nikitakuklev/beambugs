---
title: "TRK-20: Nearly cancelled bend focusing corrupts the expanded matrix map"
hide:
  - toc
---

# TRK-20 — Nearly cancelled bend focusing corrupts the expanded matrix map

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Failure | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`track_expanded_combined_dipole_quad_single_particle`, track_magnet_drift.h:119–147](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_drift.h#L119-L147)<br>[dividing rounded differences by Kₓ, lines 185–226](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_drift.h#L185-L226)<br>[vertical coefficients, lines 238–266](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_drift.h#L238-L266)<br>[model selection, track_magnet_configure.h:64–69](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_configure.h#L64-L69)<br>[dispatch, track_magnet_drift.h:579–580](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_drift.h#L579-L580)<br>[allocation, track_magnet_configure.h:128–138](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_configure.h#L128-L138)<br>[curvature kick, track_magnet_kick.h:176–187](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_kick.h#L176-L187)<br>[uniform composition, integrator.h:51–61](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/integrator.h#L51-L61)<br>[expanded-map Hamiltonian, xtrack.tex:1105–1119](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1105-L1119)<br>[effective focusing and finite transport functions, 1130–1177](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1130-L1177)<br>[transverse and nonzero-focusing path map, 1182–1217](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1182-L1217)<br>[arrival update, 1248–1253](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L1248-L1253) |

With the mat-kick-mat model, a combined-function bend near cancellation of horizontal focusing can give incorrect horizontal positions and arrival coordinates. A 1.000 m bend with curvature h = 0.2000 m⁻¹, dipole strength k₀ = 0.2000 m⁻¹, quadrupole strength k₁ = −0.04000 m⁻² and normalized horizontal momentum pₓ = 0.001000 gives arrival coordinate ζ = −110.5 µm instead of −160.5 µm. A 0.4% dipole-field mismatch with nearly cancelled focusing also gives a 122.5 µm horizontal error in a 0.7000 m bend. High-precision integration of the prescribed split map establishes the finite limiting result.

Run `python TRK-20.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-20.py"

    [Download TRK-20.py](TRK-20.py)

    ```python
    """TRK-20: Nearly cancelled bend focusing corrupts the expanded matrix map

    With the mat-kick-mat model, a combined-function bend near cancellation of horizontal focusing can give incorrect horizontal positions and arrival coordinates. A 1.000 m bend with curvature h = 0.2000 m⁻¹, dipole strength k₀ = 0.2000 m⁻¹, quadrupole strength k₁ = −0.04000 m⁻² and normalized horizontal momentum pₓ = 0.001000 gives arrival coordinate ζ = −110.5 µm instead of −160.5 µm. A 0.4% dipole-field mismatch with nearly cancelled focusing also gives a 122.5 µm horizontal error in a 0.7000 m bend. High-precision integration of the prescribed split map establishes the finite limiting result.

    Track fixed protons through the public mat-kick-mat Bend with disabled edges and the uniform integrator. Integrate its prescribed differential equations at 80 decimal digits, including both matrix fragments and the curvature kick. Compare ordinary decimal designs and neighboring gradients with exactly zero and ordinary focusing controls.
    """

    import math

    import mpmath as mp
    from _prebuilt import CTX, identity, kernels, np, particles, xt


    def coefficients(q, p, acceleration, k):
        # q'' = acceleration - k*q; a[n] multiplies s**n.
        # For these short fragments 48 coefficients exceed the 80-digit accuracy.
        a = [q, p, (acceleration - k * q) / 2]
        for n in range(1, 46):
            a.append(-k * a[n] / ((n + 1) * (n + 2)))
        return a


    def integral(a, length):
        return mp.fsum(v * length ** (i + 1) / (i + 1) for i, v in enumerate(a))


    def square_integral(a, length):
        return mp.fsum(
            ai * aj * length ** (i + j + 1) / (i + j + 1)
            for i, ai in enumerate(a)
            for j, aj in enumerate(a)
        )


    def flow(v, length, h, k0, k1):
        x, px, y, py, zeta = v
        a = coefficients(x, px, h - k0, k0 * h + k1)
        b = coefficients(y, py, mp.mpf(0), -k1)
        da = [i * a[i] for i in range(1, len(a))]
        db = [i * b[i] for i in range(1, len(b))]
        # chi=1, delta=0, rvv=1: zeta'=-h*x-(x'^2+y'^2)/2.
        # Integrate polynomial products directly, without divided trig differences.
        dzeta = (
            -h * integral(a, length) - (square_integral(da, length) + square_integral(db, length)) / 2
        )
        return [
            mp.polyval(a[::-1], length),
            mp.polyval(da[::-1], length),
            mp.polyval(b[::-1], length),
            mp.polyval(db[::-1], length),
            zeta + dzeta,
        ]


    def oracle(initial, length, h, k0, k1, kicks=1):
        length, h, k0, k1 = map(mp.mpf, (length, h, k0, k1))
        v = list(map(mp.mpf, initial))
        for _ in range(kicks):
            v = flow(v, length / (2 * kicks), h, k0, k1)
            x, _, y, _, _ = v
            # The public uniform split includes this prescribed curvature kick.
            v[1] += length / kicks * h * k1 * (-x * x + y * y / 2)
            v[3] += length / kicks * h * k1 * x * y
            v = flow(v, length / (2 * kicks), h, k0, k1)
        return v


    def case(label, initial, length, h, k0, k1, *, control=False, kicks=1):
        element = xt.Bend(
            _context=CTX,
            length=length,
            angle=length * h,
            k0=k0,
            k1=k1,
            model="mat-kick-mat",
            integrator="uniform",
            num_multipole_kicks=kicks,
            edge_entry_active=False,
            edge_exit_active=False,
        )
        p = particles(x=initial[0], px=initial[1], y=initial[2], py=initial[3], zeta=initial[4])
        element.track(p)
        assert p.state[0] == 1
        observed = np.array([getattr(p, key)[0] for key in ("x", "px", "y", "py", "zeta")])
        # Read the actual stored curvature, rather than assuming angle/length is exact.
        args = (initial, length, float(element.h), k0, k1, kicks)
        expected_mp = oracle(*args)
        with mp.workdps(100):
            refined = oracle(*args)
            assert max(abs(a - b) for a, b in zip(expected_mp, refined)) < mp.mpf("1e-70")
        expected = np.array([float(v) for v in expected_mp])
        match = bool(np.allclose(observed, expected, rtol=0, atol=1e-14))
        print(
            f"{label}: observed_x={observed[0]:.17g} expected_x={expected[0]:.17g} "
            f"observed_zeta={observed[4]:.17g} expected_zeta={expected[4]:.17g} "
            f"max_error={np.max(np.abs(observed - expected)):.17g} "
            f"derivation=80-digit ODE polynomial integration plus curvature kick match={match}"
        )
        if control:
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
        return match


    def main():
        identity()
        mp.mp.dps = 80
        print(f"ORACLE mpmath {mp.__version__}; 80 digits checked against 100; atol=1e-14 m")
        original = [0.003, 0.021, -0.017, 0.008, 0.0]
        paraxial = [0.0003, 0.001, -0.0005, 0.0002, 0.0]
        matches = []
        matches.append(case("exact_zero_control", original, 0.7, 0.25, 0.25, -0.0625, control=True))
        for kx in (-1.0, 1.0):
            matches.append(
                case(f"ordinary_Kx={kx}", original, 0.7, 0.25, 0.25, kx - 0.0625, control=True)
            )
        matches.append(case("decimal_design", paraxial, 1.0, 0.2, 0.2, -0.04))
        matches.append(case("decimal_mismatch", paraxial, 1.0, 0.2, 0.201, -0.0402))
        for direction in (-math.inf, math.inf):
            matches.append(
                case(
                    f"matched_neighbor={direction}",
                    original,
                    0.7,
                    0.25,
                    0.25,
                    math.nextafter(-0.0625, direction),
                )
            )
        matches.append(
            case("mismatch_cancelled", original, 0.7, 0.25, 0.251, math.nextafter(-0.06275, math.inf))
        )
        matches.append(
            case(
                "mismatch_32_kicks",
                original,
                0.7,
                0.25,
                0.251,
                math.nextafter(-0.06275, math.inf),
                kicks=32,
            )
        )
        matches.append(case("mismatch_only_control", original, 0.7, 0.25, 0.251, -0.0625, control=True))
        kernels()
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | <span class="bb-nowrap">`gpt-6-astra`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | <span class="bb-nowrap">`gpt-6.1-sol`</span> | xhigh | <span class="bb-nowrap">2026-11-06</span> | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

