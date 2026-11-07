---
title: "OPT-03: Solenoid Taylor map has zero strength sensitivity at zero field"
hide:
  - toc
---

# OPT-03 — Solenoid Taylor map has zero strength sensitivity at zero field

| | |
|---|---|
| Code | [Xsuite](../index.md) · Optics and matching |
| Priority | P4 |
| Severity | Silent wrong result |
| Evidence | Probe-pinned |
| Last reproduced | 2026-10-26 on xtrack release [v0.118.4](https://github.com/xsuite/xtrack/tree/v0.118.4) (`3cd8bfc`) |
| Tags | <span class="bb-tag">experimental</span> [?](../../methodology.md#tags) |
| Source | [`track_magnet_drift.h:475–482`](https://github.com/xsuite/xtrack/blob/ad5e808cdc5b9a881eac2f41e32a0b7d11cc7028/xtrack/beam_elements/elements_src/track_magnet_drift.h#L475-L482) |

Expanding a `UniformSolenoid` about zero field, with its normalized strength `ks` as a parameter of the experimental `ParticlesTpsa` API for native truncated-power-series (Taylor) maps, returns zero orbit sensitivity to that strength. Differentiating the exact solenoid map gives a nonzero response, confirmed by central differences of ordinary particle tracking. For a 0.7 m solenoid and a 1 GeV/c proton starting at x = 3 mm with all other coordinates zero, the derivative of vertical position y with respect to `ks` is 0 instead of −1.050 × 10⁻³ m². A calculation using this derivative, such as an optimizer starting from a switched-off solenoid, sees no response; tested expansions about nonzero field agree with numerical differentiation.

Run `python OPT-03.py` with [_common.py](../helpers/_common.py) in the same directory.

??? example "Reproducer: OPT-03.py"

    [Download OPT-03.py](OPT-03.py)

    ```python
    """OPT-03: Solenoid Taylor map has zero strength sensitivity at zero field

    Expanding a `UniformSolenoid` about zero field, with its normalized strength `ks` as a parameter of
    the experimental `ParticlesTpsa` API for native truncated-power-series (Taylor) maps, returns zero
    orbit sensitivity to that strength. Differentiating the exact solenoid map gives a nonzero response,
    confirmed by central differences of ordinary particle tracking. For a 0.7 m solenoid and a 1 GeV/c
    proton starting at x = 3 mm with all other coordinates zero, the derivative of vertical position y
    with respect to `ks` is 0 instead of −1.050 × 10⁻³ m². A calculation using this derivative, such as
    an optimizer starting from a switched-off solenoid, sees no response; tested expansions about
    nonzero field agree with numerical differentiation.

    The script gives a 0.7 m `UniformSolenoid` a first-order `ks` parameter with zero constant part and
    reads ∂y/∂`ks` from the `ParticlesTpsa` map of a 1 GeV/c proton at x₀ = 3 mm. The exact map y = −x₀
    cos θ sin θ with θ = `ks`·L / (2√(1 − (`ks`·x₀/2)²)) gives −L·x₀/2 at zero field; central
    differences of ordinary tracking at `ks` = ±10⁻⁵ and ±5 × 10⁻⁶ m⁻¹ must agree with it before the map
    coefficient is compared.
    """

    import madng_tpsa
    import numpy as np
    import xtrack as xt
    from xtrack.tpsa import ParticlesTpsa

    RTOL, ATOL = 2e-12, 2e-15


    def evaluate():
        length, x0 = 0.7, 0.003
        expected = -length * x0 / 2
        # From y=-x0*cos(theta)*sin(theta), theta=ks*L/(2*p_s), p_s(0)=1.
        for step in (1e-5, 5e-6):
            values = []
            for ks in (-step, step):
                particle = xt.Particles(p0c=1e9, x=x0)
                xt.UniformSolenoid(length=length, ks=ks).track(particle)
                values.append(float(particle.y[0]))
            derivative = (values[1] - values[0]) / (2 * step)
            # Central-difference truncation is O(step**2); both steps test the limit.
            np.testing.assert_allclose(derivative, expected, rtol=1e-10, atol=2e-15)
        element = xt.UniformSolenoid(length=length, ks=0)
        line = xt.Line(elements=[element])  # Parameter storage requires line ownership.
        descriptor = madng_tpsa.Descriptor(6, 1, num_params=1, param_order=1)
        # Xobjects creates the public field descriptor dynamically.
        element.ks = descriptor.param(1, 0.0)  # ty: ignore[unresolved-attribute]
        series = ParticlesTpsa(order=1, descriptor=descriptor, p0c=1e9, x=x0)
        element.track(series)
        assert series._xobject.state == 1
        return {"dy/dks at ks=0": (series.param_jacobian()[2, 0], expected)}


    if __name__ == "__main__":
        from _common import prepare, report

        prepare()
        print("Expected: dy/dks=-L*x0/2=-0.00105; checked by central differences at h=1e-5 and 5e-6.")
        report(evaluate(), rtol=RTOL, atol=ATOL)

    ```

## Provenance

| Role | Model | Effort | Date | Code examined |
|---|---|---|---|---|
| Found | `gpt-6-astra` | xhigh | 2026-10-26 | xtrack 0.118.2 (`ad5e808cdc5b`), installed wheels, prebuilt kernels, CPU, double, serial |
| Confirmed | `gpt-6.1-sol` | xhigh | 2026-10-30 | xtrack 0.118.2 (`ad5e808cdc5b`), installed packages, prebuilt kernels (compilation forbidden), CPU, double, serial |

