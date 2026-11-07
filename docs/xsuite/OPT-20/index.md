---
title: "OPT-20: Open chromatic Twiss launches pζ dispersion as a δ derivative"
hide:
  - toc
---

# OPT-20 — Open chromatic Twiss launches pζ dispersion as a δ derivative

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P2 |
| Severity | Silent wrong result |
| Evidence | Reproduced, independently rechecked |
| Last reproduced | 2026-11-07 on xtrack main (`f09e0fb`, head build) |
| Source | [_build_open_off_momentum_init, chromatic_functions.py:205–241](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/chromatic_functions.py#L205-L241)<br>[chromatic differences:56–64](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/chromatic_functions.py#L56-L64)<br>[lattice_functions_from_W.py:132–152](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/lattice_functions_from_W.py#L132-L152)<br>[twiss.py:173–176](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L173-L176)<br>[193–194](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/twiss.py#L193-L194)<br>[dispersion as dx/dpζ](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L4963-L4967)<br>[exact longitudinal coordinates](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/physics_manual/xtrack.tex#L335-L381)<br>[off-momentum and initial-condition optics](https://github.com/xsuite/xsuite/blob/e0913e0a18028b2fce06991aebed6150876124b3/docs/twiss.rst#L61-L87)<br>[periodic branch](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/twiss/chromatic_functions.py#L24-L33) |

Open chromatic Twiss at nonzero momentum offset launches its auxiliary orbits using dispersion with respect to canonical momentum pζ as though it were a derivative with respect to relative momentum offset δ. The required conversion is dpζ/dδ = β/β₀, where β and β₀ are particle and reference speeds divided by the speed of light. For 800.0 MeV/c reference protons at δ = 0.18, a sextupole channel gives dβₓ/dδ = −2.166 m instead of −2.282 m. Independent differentiation of tracked transverse Jacobians establishes the expected result in both planes. Both open Twiss modes fail; zero-offset and zero-sextupole controls pass.

Run `python OPT-20.py` with [_prebuilt.py](../helpers/_prebuilt.py) and [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-20.py"

    [Download OPT-20.py](OPT-20.py)

    ```python
    """OPT-20: Open chromatic Twiss launches pζ dispersion as a δ derivative

    Open chromatic Twiss at nonzero momentum offset launches its auxiliary orbits using dispersion with respect to canonical momentum pζ as though it were a derivative with respect to relative momentum offset δ. The required conversion is dpζ/dδ = β/β₀, where β and β₀ are particle and reference speeds divided by the speed of light. For 800.0 MeV/c reference protons at δ = 0.18, a sextupole channel gives dβₓ/dδ = −2.166 m instead of −2.282 m. Independent differentiation of tracked transverse Jacobians establishes the expected result in both planes. Both open Twiss modes fail; zero-offset and zero-sextupole controls pass.

    The script differentiates independently tracked transverse Jacobians along prescribed physical entrance orbits, then transports the entrance mode covariances. A normal and skew sextupole channel exercises both transverse planes and both open Twiss modes at two nested finite-difference resolutions. Zero-offset and zero-sextupole cases are controls; canonical input dispersion is obtained using dpζ/dδ = β/β₀.
    """

    from _prebuilt import CTX, identity, kernels, np, xt


    def close(a, b, *, atol=3e-10, rtol=3e-10):
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


    def make_line(elements, p):
        line = xt.Line(elements=elements)
        line.particle_ref = p.copy()
        line.build_tracker(_context=CTX)
        return line


    identity()
    np.set_printoptions(linewidth=200)
    failed = 0
    P0 = 800e6
    MASS = xt.PROTON_MASS_EV
    D = np.array([0.07, 0.012, -0.04, -0.008])
    W = np.zeros((4, 4))
    for j, beta, alpha in ((0, 2.4, -0.35), (2, 1.8, 0.25)):
        W[j : j + 2, j : j + 2] = [[np.sqrt(beta), 0], [-alpha / np.sqrt(beta), 1 / np.sqrt(beta)]]
    keys = ("x", "px", "y", "py")


    def tracked_optics(line, delta, displacement, inner_step):
        rays = displacement[:, None] + inner_step * np.concatenate((np.eye(4), -np.eye(4)), axis=1)
        particles = xt.Particles(_context=CTX, p0c=P0, mass0=MASS, delta=delta, **dict(zip(keys, rays)))
        line.track(particles)
        assert np.all(particles.state == 1)
        result = np.array([getattr(particles, key) for key in keys])
        matrix = (result[:, :4] - result[:, 4:]) / (2 * inner_step)
        basis = matrix @ W
        return np.array(
            [
                np.dot(basis[0, :2], basis[0, :2]),
                -np.dot(basis[0, :2], basis[1, :2]),
                np.dot(basis[2, 2:], basis[2, 2:]),
                -np.dot(basis[2, 2:], basis[3, 2:]),
            ]
        )


    for delta in (0.0, 0.015, 0.18, -0.12):
        for strength in (0.0, 5.0):
            p = xt.Particles(_context=CTX, p0c=P0, mass0=MASS, delta=delta)
            line = make_line(
                {
                    "d1": xt.Drift(length=0.2),
                    "s": xt.Multipole(knl=[0, 0, strength], ksl=[0, 0, strength * 0.26]),
                    "d2": xt.Drift(length=0.7),
                },
                p,
            )
            rvv = ((P0 * (1 + delta)) / np.hypot(P0 * (1 + delta), MASS)) / (P0 / np.hypot(P0, MASS))
            native_D = D / rvv
            for h, inner in ((2e-5, 1e-6), (1e-5, 5e-7)):
                plus = tracked_optics(line, delta + h, D * h, inner)
                minus = tracked_optics(line, delta - h, -D * h, inner)
                expected = (plus - minus) / (2 * h)
                source_plus = tracked_optics(line, delta + h, native_D * h, inner)
                source_minus = tracked_optics(line, delta - h, -native_D * h, inner)
                source_prediction = (source_plus - source_minus) / (2 * h)
                inputs = dict(
                    betx=2.4,
                    alfx=-0.35,
                    bety=1.8,
                    alfy=0.25,
                    delta=delta,
                    **dict(zip(("dx", "dpx", "dy", "dpy"), native_D)),
                    chrom=True,
                    delta_chrom=h,
                )
                tw = line.twiss(method="4d", **inputs)
                tw6 = line.twiss(method="6d", **inputs)
                for field in (
                    "betx",
                    "alfx",
                    "bety",
                    "alfy",
                    "bx_chrom",
                    "ax_chrom",
                    "by_chrom",
                    "ay_chrom",
                ):
                    close(getattr(tw6, field), getattr(tw, field), atol=2e-11, rtol=2e-11)
                result = np.array(
                    [
                        tw.bx_chrom[-1] * tw.betx[-1],
                        tw.ax_chrom[-1] + tw.bx_chrom[-1] * tw.alfx[-1],
                        tw.by_chrom[-1] * tw.bety[-1],
                        tw.ay_chrom[-1] + tw.by_chrom[-1] * tw.alfy[-1],
                    ]
                )
                ok = np.allclose(result, expected, atol=8e-9, rtol=8e-9)
                if not ok:
                    close(result, source_prediction, atol=8e-9, rtol=8e-9)
                if delta == 0 or strength == 0:
                    close(result, expected, atol=8e-9, rtol=8e-9)
                else:
                    failed += not ok
                center = tracked_optics(line, delta, np.zeros(4), inner)
                close(
                    center, [tw.betx[-1], tw.alfx[-1], tw.bety[-1], tw.alfy[-1]], atol=3e-10, rtol=3e-10
                )
                print(
                    f"delta={delta:g} strength={strength:g} h={h:g} inner={inner:g} modes=4d,6d: observed={result} expected={expected} beta_ratio={rvv:.15g} derivation=central_delta_difference_of_tracked_mode_covariances physical_match={ok}"
                )
    kernels()
    print("DEFECT REPRODUCED" if failed else "NOT REPRODUCED")

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-11-06 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

