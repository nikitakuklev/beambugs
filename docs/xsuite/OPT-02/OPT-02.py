"""OPT-02: Native Taylor maps ignore an absolute-time cavity's RF clock

Tracking a `Cavity` with `absolute_time=True` through the experimental `ParticlesTpsa` API, which
propagates native truncated-power-series (Taylor) maps, silently ignores the absolute RF clock and
so uses the wrong RF phase. The absolute-arrival-time definition and the relativistic
energy–momentum relation give the correct kick, and ordinary particle tracking reproduces it. For a
1 GeV/c proton at zero position and RF phase, a 1 MV, 100 MHz cavity with `time_s` = 2.5 ns gives a
relative momentum change δ of zero instead of 1.371 × 10⁻³, a 1.000 MeV energy gain; relative-time
controls are correct. A source comment acknowledges the restriction, but the public interface
accepts the configuration without warning or rejection.

The script tracks a 1 GeV/c proton at a quarter RF period (`time_s` = 2.5 ns) through a 1 MV, 100
MHz `Cavity` at zero phase, as an ordinary particle and as a first-order `ParticlesTpsa` map. With
`absolute_time=True` the RF phase is π/2, so the proton must gain exactly 1 MeV; ordinary tracking
is checked against that before the map's constant part is compared. With `absolute_time=False` both
give zero, which isolates the clock term.
"""

import numpy as np
import xtrack as xt
from xtrack.tpsa import ParticlesTpsa

RTOL, ATOL = 2e-13, 2e-16


def evaluate():
    frequency, voltage, p0c = 1e8, 1e6, 1e9
    cases = {}
    for absolute_time in (False, True):
        element = xt.Cavity(
            voltage=voltage, frequency=frequency, phase=0, absolute_time=absolute_time
        )
        kwargs = {"p0c": p0c, "time_s": 0.25 / frequency, "s": 0, "zeta": 0}
        scalar = xt.Particles(**kwargs)
        series = ParticlesTpsa(order=1, p0c=p0c, time_s=0.25 / frequency, s=0, zeta=0)
        # At t=T/4 the absolute RF phase is pi/2; at relative phase zero it is zero.
        ptau = voltage / p0c if absolute_time else 0.0
        beta0 = float(scalar.beta0[0])
        u = ptau * (ptau + 2 / beta0)
        expected = u / (np.sqrt(1 + u) + 1)
        element.track(scalar)
        np.testing.assert_allclose(scalar.delta[0], expected, rtol=RTOL, atol=ATOL)
        element.track(series)
        assert series._xobject.state == 1
        if not absolute_time:
            np.testing.assert_allclose(series.const_part[5], expected, rtol=RTOL, atol=ATOL)
        cases[f"Cavity absolute_time={absolute_time}, TPSA delta"] = series.const_part[5], expected
    return cases


if __name__ == "__main__":
    from _common import prepare, report

    prepare()
    print(
        "Expected: at absolute time T_RF/4, Delta E=V=1e6 eV; relative-time control has zero kick."
    )
    report(evaluate(), rtol=RTOL, atol=ATOL)
