"""OPT-04: Finite-difference transfer matrices reset the supplied RF clock

The finite-difference transfer matrix (`get_R_matrix`) and open Twiss propagation drop the supplied
orbit's clock `time_s`, resetting the RF clock of an absolute-time cavity. Shifting the clock while
compensating with the cavity phase leaves tracking and its derivatives unchanged, as the analytic RF
focusing and central differences that keep the full particle state confirm. For a 1 GeV/c proton and
a 1 MV, 100 MHz cavity with a compensated clock phase of π/3 rad, the longitudinal focusing term,
the change of momentum coordinate `pzeta` per longitudinal offset `zeta`, is −1.970 × 10⁻³ m⁻¹
instead of −3.941 × 10⁻³ m⁻¹, and open Twiss generates `pzeta` = −1.188 × 10⁻³ instead of zero.
Supplied nonzero clocks are affected; zero-clock controls pass.

The script builds a ring of one absolute-time `Cavity` (1 MV, 100 MHz, phase equal to minus the
clock phase) and a drift of one RF wavelength, and checks that a 1 GeV/c proton orbit with clock
phase 0 or π/3 closes. It compares the `get_R_matrix` element ∂`pzeta`/∂`zeta` with the analytic
−(V/p0c)·2πf/(c·β₀²), where β₀ is the reference speed over c, and with central differences of
tracked copies of the full orbit particle, and checks that open Twiss started from the orbit keeps
`pzeta` = 0. The zero-clock case is the control.
"""

import numpy as np
import xtrack as xt
from scipy.constants import c

RTOL, ATOL = 2e-9, 2e-12


def evaluate():
    voltage, frequency, p0c = 1e6, 1e8, 1e9
    cases = {}
    for clock_phase in (0.0, np.pi / 3):
        # Compensating phase puts the supplied orbit at a zero-voltage crossing.
        reference = xt.Particles(p0c=p0c, time_s=clock_phase / (2 * np.pi * frequency))
        beta0 = float(reference.beta0[0])
        # One ring revolution is exactly one RF period, so this is a periodic orbit.
        line = xt.Line(
            elements={
                "rf": xt.Cavity(
                    voltage=voltage, frequency=frequency, phase=-clock_phase, absolute_time=True
                ),
                "d": xt.Drift(length=beta0 * c / frequency),
            }
        )
        line.particle_ref = reference.copy()
        line.build_tracker()
        orbit = reference.copy()
        line.track(orbit)
        np.testing.assert_allclose([orbit.zeta[0], orbit.pzeta[0]], [0, 0], rtol=0, atol=1e-14)
        expected = -voltage / p0c * (2 * np.pi * frequency / c) / beta0**2
        # An independent central difference copies the entire reference state.
        h = 2e-6
        samples = []
        for zeta in (-h, h):
            probe = reference.copy()
            probe.zeta = zeta
            line.track(probe)
            samples.append(float(probe.pzeta[0]))
        direct = (samples[1] - samples[0]) / (2 * h)
        np.testing.assert_allclose(direct, expected, rtol=RTOL, atol=ATOL)
        observed = line.get_R_matrix(particle_on_co=reference, steps={"dzeta": h})["R_matrix"][5, 4]
        if clock_phase == 0:
            np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)
        cases[f"clock phase={clock_phase}, R[pzeta,zeta]"] = observed, expected
        init = xt.TwissInit(particle_on_co=reference.copy(), W_matrix=np.eye(6), element_name="rf")
        tw = line.twiss(init=init, chrom=False)
        if clock_phase == 0:
            np.testing.assert_allclose(tw.pzeta, 0, rtol=0, atol=1e-14)
        cases[f"clock phase={clock_phase}, open Twiss pzeta"] = tw.pzeta, np.zeros(3)
    return cases


if __name__ == "__main__":
    from _common import prepare, report

    prepare()
    print("Expected: -V/p0c*2*pi*f/(c*beta0**2), invariant under a simultaneous clock/phase shift.")
    print("The orbit closes; independent tracking differences preserve its clock.")
    report(evaluate(), rtol=RTOL, atol=ATOL)
