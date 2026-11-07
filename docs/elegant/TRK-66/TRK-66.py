"""TRK-66: CWIGGLER field normalization treats reference momentum as Lorentz gamma

A CWIGGLER specified by magnetic field uses √(P₀²−1) instead of P₀ for its reference rigidity, where P₀ is reference momentum in units mc. This makes the physical field and trajectory depend on the chosen reference. For a 0.8000 m, 0.03000 T sinusoidal wiggler at 30.00 mc, the paraxial path excess is 69.83 µm instead of 69.75 µm. Independent integration of the magnetic impulse and direct evaluation of both field routines establish the misplaced energy conversion. The field excess falls to 1.451×10⁻⁸ at 3 GeV, making the practical concern low-energy use.

The script integrates the physical sinusoidal magnetic impulse and its paraxial path excess, using CWIGGLER's own electron constants. It tracks the same 30 mc particle at three reference momenta, two integrator orders and two resolutions, with radiation disabled and a zero-field control. A reference-only path difference and the predicted erroneous rigidity factor isolate the normalization.
"""

import numpy as np
from _cwiggler import FIELD_RIGIDITY, C, close, track
from _elegant import verdict


def main():
    length, period, field, pin = 0.8, 0.2, 0.03, 30.0
    # Integrate Px/(mc)=-e*B*sin(kz)/(mc*k), k=2*pi/period.
    # The implemented paraxial Hamiltonian has ds/dz=1+(Px/P)^2/2;
    # an integer number of periods gives L*(e*B/(mc*k*P))^2/4.
    expected = length / 4 * (FIELD_RIGIDITY * field * period / (2 * np.pi * pin)) ** 2
    defect = False
    paths = {}
    for order, steps in ((2, 32), (4, 32), (4, 64)):
        for p0 in (30.0, 45.0, 15.0):
            lattice = (
                f"W:CWIGGLER,L={length},PERIODS=4,BMAX={field},SINUSOIDAL=1,"
                f"INTEGRATION_ORDER={order},STEPS_PER_PERIOD={steps},"
                "SYNCH_RAD=0,ISR=0\nBL:LINE=(W)\n"
            )
            out = track(lattice, p0, [[0, 0, 0, 0, 0, pin]])
            observed = float(C * out["t"][0] * pin / np.hypot(pin, 1) - length)
            close(out["p"], pin, 2e-14, "radiation-off momentum")
            # E0=p0*mc^2 passed as gamma yields f^2=p0^2/(p0^2-1).
            faulty = expected * p0**2 / (p0**2 - 1)
            if abs(observed - expected) > 4e-14:
                close(observed, faulty, 4e-14, "incorrect rigidity factor")
                defect = True
            paths[order, steps, p0] = observed
            print(
                f"serial/order={order} steps={steps} p0={p0:g} mc: "
                f"path_excess={observed:.17g} m expected={expected:.17g} m "
                f"(integrated Lorentz impulse and paraxial path); faulty={faulty:.17g} m"
            )
    difference = paths[4, 64, 30.0] - paths[4, 64, 45.0]
    print(
        f"serial/reference 30->45 mc: path_difference={difference:.17g} m expected=0 (same physical beam and field)"
    )
    out = track(
        "W:CWIGGLER,L=0.8,PERIODS=4,BMAX=0,SINUSOIDAL=1,SYNCH_RAD=0,ISR=0\nBL:LINE=(W)\n",
        pin,
        [[0, 0, 0, 0, 0, pin]],
    )
    zero = float(C * out["t"][0] * pin / np.hypot(pin, 1) - length)
    close(zero, 0, 4e-14, "zero-field path")
    close(out["p"], pin, 2e-14, "zero-field momentum")
    close(out["xp"], 0, 4e-14, "zero-field horizontal slope")
    close(out["yp"], 0, 4e-14, "zero-field vertical slope")
    print(f"serial/zero-field: path_excess={zero:.17g} m expected=0 (straight path)")
    verdict(defect)


if __name__ == "__main__":
    main()
