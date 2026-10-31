"""TRK-01: Solenoid body ignores per-particle charge-to-mass scaling

Xtrack's `UniformSolenoid` and a constant-profile `VariableSolenoid` move every particle along the
reference-species trajectory, ignoring its charge-to-mass ratio relative to the reference species,
χ. Because the solenoid strength `ks` is normalized to the reference rigidity, a particle with ratio
χ must see χ·`ks`; the exact solenoid map and an independent integration of the Lorentz force both
require this. A 1 GeV/c particle with twice the reference charge and the same mass (χ = 2) entering
a 0.7 m solenoid with `ks` = 0.8 m⁻¹ at x = 3 mm exits at y = −0.7968 mm instead of −1.350 mm. Beams
that mix species or charge states are affected; reference-species particles are tracked correctly.

The script tracks three particles of equal mass with χ = 0.5, 1 and 2 from x = 3 mm through a 0.7 m
`UniformSolenoid` and a constant-profile `VariableSolenoid` (`ks` = 0.8 m⁻¹, p0c = 1 GeV) and
compares each exit position and momentum with the exact hard-edge solenoid map evaluated with
χ·`ks`. The χ = 1 particle must agree and checks signs and conventions; the other two show the bug.
"""

import numpy as np
import xtrack as xt

RTOL, ATOL = 2e-13, 2e-16


def evaluate():
    length, ks, x0 = 0.7, 0.8, 0.003
    chi = np.array([0.5, 1.0, 2.0])
    # Exact hard-edge solenoid map for a particle starting at x0 with zero momenta:
    # k = chi*ks/2, p_s = sqrt(1 - (k*x0)^2), theta = k*L/p_s, c, s = cos(theta), sin(theta);
    # (x, px, y, py) = (x0 c^2, -k x0 c s, -x0 c s, k x0 s^2) and zeta = L*(1 - 1/p_s).
    k = chi * ks / 2
    ps = np.sqrt(1 - (k * x0) ** 2)
    theta = k * length / ps
    c, s = np.cos(theta), np.sin(theta)
    expected = np.array(
        [x0 * c * c, -k * x0 * c * s, -x0 * c * s, k * x0 * s * s, length * (1 - 1 / ps)]
    ).T
    cases = {}
    for cls, kwargs in (
        (xt.UniformSolenoid, {"ks": ks}),
        (xt.VariableSolenoid, {"ks_profile": [ks, ks]}),
    ):
        element = cls(length=length, **kwargs)
        particles = xt.Particles(p0c=1e9, x=x0, delta=0, mass_ratio=1, charge_ratio=chi)
        element.track(particles)
        assert np.all(particles.state == 1)
        observed = np.array(
            [particles.x, particles.px, particles.y, particles.py, particles.zeta]
        ).T
        # The reference species checks signs, fringe convention and exact kinematics.
        np.testing.assert_allclose(observed[1], expected[1], rtol=RTOL, atol=ATOL)
        cases[f"{cls.__name__}, chi=0.5/1/2, columns=x,px,y,py,zeta"] = observed, expected
    return cases


if __name__ == "__main__":
    from _common import prepare, report

    prepare()
    print("Expected: exact solenoid map with k=chi*ks/2, theta=k*L/p_s.")
    report(evaluate(), rtol=RTOL, atol=ATOL)
