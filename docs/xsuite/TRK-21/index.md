---
title: "TRK-21: Dipole wedge returns NaNs for valid near-grazing momenta"
hide:
  - toc
---

# TRK-21 — Dipole wedge returns NaNs for valid near-grazing momenta

| | |
|---|---|
| Code | [Xsuite](../index.md) · Tracking |
| Priority | P3 |
| Failure | Crash or API hazard |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack release [v0.118.6](https://github.com/xsuite/xtrack/tree/v0.118.6) (`f09e0fb`) |
| Tags | <span class="bb-tag">extreme-regime</span> <span class="bb-tag">partly-source-only</span> [?](../../methodology.md#tags) |
| Source | [`Wedge_single_particle`, track_wedge.h:23–51](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_wedge.h#L23-L51)<br>[public Wedge API, wedge.py:9–23](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/wedge.py#L9-L23)<br>[both orderings, wedge.h:23–32](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/wedge.h#L23-L32)<br>[dipole-wedge map, xtrack.tex:2754–2782](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L2754-L2782)<br>[track_wedge.h:92–97](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_wedge.h#L92-L97) |

A dipole wedge returns NaN coordinates for a valid particle almost tangent to its entrance plane. For wedge angle −0.1000 rad and normalized dipole strength 0.2500 m⁻¹, rounding pushes an inverse-sine argument above one, although the stored momenta give positive squared longitudinal momentum. High-precision circle-plane geometry gives a finite 6.805 mm flight path and arrival coordinate ζ = −6.805 mm. The particle remains marked alive, and reversing the horizontal coordinates, field and wedge angle produces the same failure.

Run `python TRK-21.py` with [_prebuilt.py](../helpers/_prebuilt.py), [_wedge_geometry.py](../helpers/_wedge_geometry.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: TRK-21.py"

    [Download TRK-21.py](TRK-21.py)

    ```python
    """TRK-21: Dipole wedge returns NaNs for valid near-grazing momenta

    A dipole wedge returns NaN coordinates for a valid particle almost tangent to its entrance plane. For wedge angle −0.1000 rad and normalized dipole strength 0.2500 m⁻¹, rounding pushes an inverse-sine argument above one, although the stored momenta give positive squared longitudinal momentum. High-precision circle-plane geometry gives a finite 6.805 mm flight path and arrival coordinate ζ = −6.805 mm. The particle remains marked alive, and reversing the horizontal coordinates, field and wedge angle produces the same failure.

    Track fixed valid protons through both public Wedge map orderings and their horizontal reflections. Solve the Lorentz circle intersection with the rotated plane at 80 decimal digits using a rationalized half-angle quadratic. Check ordinary momenta and distinguish the adjacent valid and invalid stored inputs.
    """

    import math

    import mpmath as mp
    from _prebuilt import CTX, identity, kernels, np, particles, xt
    from _wedge_geometry import geometry


    def case(label, sign, px, py, order, *, control=False):
        initial = [-sign * 0.003, sign * px, 0.0, py, 0.0]
        theta, k = -sign * 0.1, sign * 0.25
        expected_mp, (pz2, flight, residual, outgoing_pz) = geometry(theta, k, initial)
        assert pz2 > 0 and flight > 0 and outgoing_pz > 0
        assert abs(residual) < mp.mpf("1e-70")
        with mp.workdps(100):
            refined, _ = geometry(theta, k, initial)
            assert max(abs(a - b) for a, b in zip(expected_mp, refined)) < mp.mpf("1e-70")
        p = particles(x=initial[0], px=initial[1], y=0, py=py, zeta=0)
        xt.Wedge(_context=CTX, angle=theta, k=k, k1=0, quad_wedge_then_dip_wedge=order).track(p)
        observed = np.array([getattr(p, key)[0] for key in ("x", "px", "y", "py", "zeta")])
        expected = np.array([float(v) for v in expected_mp])
        # The defect is a NaN at admissible input. Adjacent finite errors are
        # assessed separately because the geometry is sensitive near grazing.
        finite = bool(np.all(np.isfinite(observed)))
        print(
            f"{label} sign={sign} order={order}: observed={observed.tolist()} "
            f"expected={expected.tolist()} state={int(p.state[0])} "
            f"pz2={float(pz2):.17g} flight={float(flight):.17g} "
            f"asin_argument={(1 / math.sqrt(1 - py * py)) * initial[1]:.17g} "
            f"derivation=80-digit Lorentz circle-plane quadratic finite={finite}"
        )
        assert p.state[0] == 1
        if control:
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-14)
        elif label == "inside_neighbor":
            assert finite
            np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-9)
        return finite


    def main():
        identity()
        mp.mp.dps = 80
        print(f"ORACLE mpmath {mp.__version__}; 80 digits checked against 100")
        px, py = 0.4371511611108554, 0.8993880488084278
        matches = []
        for sign in (1, -1):
            for order in (0, 1):
                matches.append(case("grazing", sign, px, py, order))
            case("ordinary_control", sign, 0.4, 0.8, 0, control=True)
            case("inside_neighbor", sign, math.nextafter(px, 0), py, 0)
            # Exclude genuinely invalid input instead of silently clipping its domain.
            outside = math.nextafter(px, math.inf)
            pz2 = 1 - mp.mpf(outside) ** 2 - mp.mpf(py) ** 2
            assert pz2 < 0
            print(
                f"outside_neighbor sign={sign}: exact_pz2={float(pz2):.17g} "
                "expected=invalid derivation=1-px^2-py^2; excluded from tracking"
            )
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

