"""OPT-01: Native Taylor maps silently omit RFMultipole and CrabCavity kicks

Tracking an `RFMultipole` or `CrabCavity` with the experimental `ParticlesTpsa` API, which
propagates native truncated-power-series (Taylor) maps, silently drops the element's transverse kick
and the energy change that goes with it. The RF potential predicts both kicks, and ordinary particle
tracking reproduces them. For a 1 GeV/c proton at x = 3 mm crossing a 1 MV, 100 MHz crab cavity at
phase π/4 rad, the map gives zero normalized horizontal momentum pₓ and relative momentum change δ
instead of 7.071 × 10⁻⁴ and −6.097 × 10⁻⁶; an RF dipole with `knl[0]` = 0.001 likewise loses its
opposite kick. Ordinary finite-difference `Line.twiss()` is not shown to fail.

The script tracks a 1 GeV/c proton at x = 3 mm through an `RFMultipole` (`knl[0]` = 0.001) and a 1
MV `CrabCavity`, both at 100 MHz and phase π/4, once as an ordinary particle and once as a
first-order `ParticlesTpsa` map. The ordinary particle must first match the analytic kick of the
order-zero RF potential, with δ from the exact relativistic conversion of the energy change; the
constant part of the map is then compared with the same values.
"""

import numpy as np
import xtrack as xt
from scipy.constants import c
from xtrack.tpsa import ParticlesTpsa

RTOL, ATOL = 2e-13, 2e-16


def evaluate():
    p0c, x, frequency = 1e9, 0.003, 1e8
    phase = np.pi / 4
    # Both chosen order-zero potentials give the same longitudinal kick.
    ptau = -(2 * np.pi * frequency / c) * 0.001 * x * np.sin(phase)
    beta0 = float(xt.Particles(p0c=p0c).beta0[0])
    radicand_change = ptau * (ptau + 2 / beta0)
    delta = radicand_change / (np.sqrt(1 + radicand_change) + 1)
    cases = {}
    # Order-zero transverse kicks: px = -knl[0]*cos(phi) for the RF dipole and
    # (V/p0c)*sin(phi) for the horizontal crab cavity, both 0.001 in magnitude here.
    for element, px in (
        (xt.RFMultipole(knl=[0.001], phase_n=[phase], frequency=frequency), -0.001 * np.cos(phase)),
        (xt.CrabCavity(crab_voltage=1e6, phase=phase, frequency=frequency), 0.001 * np.sin(phase)),
    ):
        expected = np.array([px, delta])
        scalar = xt.Particles(p0c=p0c, x=x)
        series = ParticlesTpsa(order=1, p0c=p0c, x=x)
        element.track(scalar)
        np.testing.assert_allclose([scalar.px[0], scalar.delta[0]], expected, rtol=RTOL, atol=ATOL)
        element.track(series)
        assert series._xobject.state == 1
        cases[f"{type(element).__name__} TPSA px,delta"] = series.const_part[[1, 5]], expected
    return cases


if __name__ == "__main__":
    from _common import prepare, report

    prepare()
    print("Expected: RF dipole potential, including its longitudinal energy derivative.")
    print("Scalar controls must match the analytic kick before testing the TPSA constant part.")
    report(evaluate(), rtol=RTOL, atol=ATOL)
