"""High-precision local forward circle-plane intersection for dipole-wedge checks."""

import mpmath as mp


def geometry(theta, k, initial):
    theta, k = mp.mpf(theta), mp.mpf(k)
    x, px, y, py, zeta = map(mp.mpf, initial)
    pz2 = 1 - px * px - py * py
    assert pz2 >= 0, "The stored particle must have real longitudinal momentum"
    pz = mp.sqrt(pz2)
    ct, st = mp.cos(theta), mp.sin(theta)
    u, v = px * ct + pz * st, pz * ct - px * st
    assert v > 0, "Select the local forward intersection"
    if k == 0:
        flight = x * st / v
        xx, zz = x + px * flight, pz * flight
        expected = [xx * ct + zz * st, u, y + py * flight, py, zeta - flight]
        return expected, (pz2, flight, zz * ct - xx * st, v)
    # The Lorentz orbit meets Z*cos(theta)-X*sin(theta)=0.
    # With t=tan(phi/2), (E+U)*t^2+2*V*t+(E-U)=0.
    # Form E-U directly and rationalize the small root to retain weak-field motion.
    e_minus_u = -k * x * st
    e_plus_u = 2 * u + e_minus_u
    radical = v * v - e_plus_u * e_minus_u
    assert radical > 0
    t = -e_minus_u / (v + mp.sqrt(radical))
    phi = 2 * mp.atan(t)
    c, s = mp.cos(phi), mp.sin(phi)
    px_lab, pz_lab = px * c - pz * s, pz * c + px * s
    xx = x + (-2 * pz * mp.sin(phi / 2) ** 2 + px * s) / k
    zz = (2 * px * mp.sin(phi / 2) ** 2 + pz * s) / k
    flight = phi / k
    # chi=1, delta=0, rvv=1 and zero reference length: Delta zeta=-flight.
    expected = [xx * ct + zz * st, px_lab * ct + pz_lab * st, y + py * flight, py, zeta - flight]
    return expected, (pz2, flight, zz * ct - xx * st, pz_lab * ct - px_lab * st)
