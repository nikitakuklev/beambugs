"""TRK-09: Solenoid spin tracking uses the wrong kinetic momentum

Spin tracking in the body of a `UniformSolenoid` takes the particle direction from the canonical
momentum plus the vector potential (`px + ax`, `py + ay`) instead of their difference, the kinetic
momentum. For motion along the field, the Thomas–BMT spin-precession equation gives a pure rotation
about the field axis and no longitudinal spin. A 1 GeV/c proton (anomalous magnetic moment G =
1.793) entering a 0.7 m solenoid with `ks` = 0.8 m⁻¹ at (x, y) = (50, 20) mm, with zero transverse
kinetic momentum and spin along x, exits with longitudinal spin −0.007126 instead of zero. Edge maps
are off, and sixteen body pieces give the same result; orbit, spin length and the G = 0 case are
correct.

The script gives the proton canonical momenta equal to the vector potential, so that it moves
straight along the axis, and tracks it through the body-only solenoid in one piece and in sixteen,
for G = 0 and G = 1.793. It compares the spin with the exact rotation about the axis by φ =
−`ks`·L·(1 + G) and checks that position, kinetic momentum and spin length are unchanged. The G = 0
case must agree and checks field normalization, rotation sign and path length; G = 1.793 shows the
bug.
"""

import numpy as np
import xtrack as xt

RTOL, ATOL = 2e-13, 2e-14


def evaluate():
    ks, length, x, y = 0.8, 0.7, 0.05, 0.02
    cases = {}
    for anomaly in (0.0, 1.79284734463):
        # With velocity parallel to B, BMT is exactly a rotation about z.
        phase = -ks * length * (1 + anomaly)
        expected = np.array([np.cos(phase), np.sin(phase), 0])
        for pieces in (1, 16):
            element = xt.UniformSolenoid(
                length=length / pieces, ks=ks, edge_entry_active=False, edge_exit_active=False
            )
            particle = xt.Particles(
                p0c=1e9,
                x=x,
                y=y,
                px=-ks * y / 2,
                py=ks * x / 2,
                ax=-ks * y / 2,
                ay=ks * x / 2,
                spin_x=1,
                anomalous_magnetic_moment=anomaly,
            )
            # Canonical momentum equals vector potential: mechanical slopes are zero.
            np.testing.assert_allclose(
                [particle.kin_px[0], particle.kin_py[0]], [0, 0], rtol=0, atol=2e-15
            )
            for _ in range(pieces):
                element.track(particle)
            assert particle.state[0] == 1
            np.testing.assert_allclose(
                [
                    np.asarray(particle.x)[0],
                    np.asarray(particle.y)[0],
                    particle.kin_px[0],
                    particle.kin_py[0],
                ],
                [x, y, 0, 0],
                rtol=0,
                atol=2e-15,
            )
            observed = np.array(
                [
                    np.asarray(getattr(particle, field))[0]
                    for field in ("spin_x", "spin_y", "spin_z")
                ]
            )
            np.testing.assert_allclose(np.linalg.norm(observed), 1, rtol=0, atol=2e-14)
            # At G = 0 the parallel and perpendicular BMT weights coincide, so the
            # direction error cancels; this control checks field normalization,
            # rotation sign and path length.
            if anomaly == 0:
                np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
            cases[f"G={anomaly}, {pieces} body pieces, spin x,y,z"] = observed, expected
    return cases


if __name__ == "__main__":
    from _common import prepare, report

    prepare()
    print("Spin enabled, radiation off; body-only solenoid, no fringe spin model involved.")
    print(
        "Expected: kinetic momentum parallel to B gives spin rotation -ks*L*(1+G) about z, with spin_z=0."
    )
    report(evaluate(), rtol=RTOL, atol=ATOL)
