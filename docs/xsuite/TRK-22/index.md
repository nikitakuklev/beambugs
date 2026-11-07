---
title: "TRK-22: Weak-field dipole wedge adds a spurious path at the rotation origin"
hide:
  - toc
---

# TRK-22 — Weak-field dipole wedge adds a spurious path at the rotation origin

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P4 |
| Severity | Wrong physics |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Tags | <span class="bb-tag">extreme-regime</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Pull request | [[gh:https://github.com/xsuite/xtrack/pull/911]], fix |
| Source | [`Wedge_single_particle`, track_wedge.h:23–51](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_wedge.h#L23-L51)<br>[public Wedge API, wedge.py:9–23](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/wedge.py#L9-L23)<br>[both orderings, wedge.h:23–32](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/wedge.h#L23-L32)<br>[dipole-wedge map, xtrack.tex:2754–2782](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L2754-L2782) |

A weak-field dipole wedge adds flight time and vertical displacement to a particle at the plane-rotation origin. Both observation planes contain this point, so the exact path length and both increments are zero. At wedge angle 0.001000 rad, normalized dipole strength 1.000×10⁻⁹ m⁻¹ and normalized momenta pₓ = 0.2000, pᵧ = 0.3000, tracking gives arrival coordinate ζ = −26.89 nm and vertical displacement 8.066 nm. Direct inverse-sine subtraction amplifies rounding by the inverse field strength; zero field and ordinary finite-field controls agree with the geometric invariant.

Run `python TRK-22.py` with [_prebuilt.py](../helpers/_prebuilt.py), [_wedge_geometry.py](../helpers/_wedge_geometry.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-22.py"

    [Download TRK-22.py](TRK-22.py)

    ```python
    """TRK-22: Weak-field dipole wedge adds a spurious path at the rotation origin

    A weak-field dipole wedge adds flight time and vertical displacement to a particle at the plane-rotation origin. Both observation planes contain this point, so the exact path length and both increments are zero. At wedge angle 0.001000 rad, normalized dipole strength 1.000×10⁻⁹ m⁻¹ and normalized momenta pₓ = 0.2000, pᵧ = 0.3000, tracking gives arrival coordinate ζ = −26.89 nm and vertical displacement 8.066 nm. Direct inverse-sine subtraction amplifies rounding by the inverse field strength; zero field and ordinary finite-field controls agree with the geometric invariant.

    Track fixed protons at the common origin of the two observation planes while varying the field and angle signs. Compare flight time and vertical motion with the exact zero-path invariant and an 80-digit circle-plane intersection. Include the strict field-cutoff neighbors, ordinary fields and an off-origin control.
    """

    import math

    import mpmath as mp
    from _prebuilt import CTX, identity, kernels, np, particles, xt
    from _wedge_geometry import geometry


    def case(label, theta, k, initial, *, control=False):
        expected_mp, (_, flight, residual, _) = geometry(theta, k, initial)
        assert abs(residual) < mp.mpf("1e-70")
        if initial[0] == 0:
            # x=0 is on both observation planes, so phi=flight=0 exactly.
            assert flight == 0 and expected_mp[2] == initial[2] and expected_mp[4] == initial[4]
        p = particles(x=initial[0], px=initial[1], y=initial[2], py=initial[3], zeta=initial[4])
        xt.Wedge(_context=CTX, angle=theta, k=k, k1=0).track(p)
        assert p.state[0] == 1
        observed = np.array([getattr(p, key)[0] for key in ("x", "px", "y", "py", "zeta")])
        expected = np.array([float(v) for v in expected_mp])
        assert np.all(np.isfinite(observed))
        match = bool(np.allclose(observed, expected, rtol=0, atol=1e-14))
        print(
            f"{label} theta={theta:.17g} k={k:.17g}: "
            f"observed_y={observed[2]:.17g} expected_y={expected[2]:.17g} "
            f"observed_zeta={observed[4]:.17g} expected_zeta={expected[4]:.17g} "
            f"derivation=zero-path invariant at x=0; circle-plane geometry elsewhere match={match}"
        )
        if control:
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
        return match


    def main():
        identity()
        mp.mp.dps = 80
        print(f"ORACLE mpmath {mp.__version__}; exact origin invariant; atol=1e-14 m")
        origin = [0.0, 0.2, 0.0, 0.3, 0.0]
        matches = []
        for theta in (0.001, -0.001, 1e-12):
            for k in (1e-9, -1e-9):
                matches.append(case("weak_origin", theta, k, origin))
        for k in (math.nextafter(1e-9, 0), 0.0, 0.25):
            matches.append(case("origin_control", 0.001, k, origin, control=True))
        small_slopes = [0.0, 0.001, 0.0, 0.0002, 0.0]
        matches.append(case("small_slope_origin", 0.001, 1e-9, small_slopes))
        case("off_origin_control", 0.001, 0.25, [0.003, 0.001, -0.002, 0.0002, 0.0], control=True)
        kernels()
        print("NOT REPRODUCED" if all(matches) else "DEFECT REPRODUCED")


    if __name__ == "__main__":
        main()

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

