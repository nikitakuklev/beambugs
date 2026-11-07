"""TRK-65: CWIGGLER radiation uses reference-normalized loss for transverse recoil

Enabling radiation in a CWIGGLER gives particles an artificial transverse recoil when their momentum differs from the reference. For a reference-normalized momentum change Δδ, the code multiplies transverse momentum by 1 + Δδ instead of 1 + Δδ/(1 + δ), where δ is the incoming fractional momentum deviation. In a 6.000 m, 1.600 T sinusoidal wiggler at 3.066 GeV/c, a +1% momentum offset produces 0.2976 ppm artificial damping of the transverse angular response. Collinear recoil requires that response to remain unchanged. Initially on-reference particles acquire a second-order error as radiation changes their momentum inside the element.

The script tracks opposite horizontal slopes at fixed physical momentum through a planar wiggler with radiation on and off, changing only the reference momentum. Collinear recoil must preserve their angular separation, and a continuum integration of the faulty normalization predicts its observed change. Both integrator orders, step refinement and initially on-reference particles are checked in fresh serial processes.
"""

import numpy as np
from _cwiggler import close, track
from _elegant import verdict


def main():
    pin, length, period, field, slope = 6000.0, 6.0, 0.1, 1.6, 0.0002
    defect = False
    for order, steps in ((2, 64), (4, 64), (4, 128)):
        for delta in (-0.01, 0.0, 0.01, 0.05):
            p0 = pin / (1 + delta)
            for sr in (0, 1):
                beam = [[0, -slope, 0, 0, 0, pin], [0, slope, 0, 0, 0, pin]]
                lattice = (
                    f"W:CWIGGLER,L={length},PERIODS={round(length / period)},"
                    f"BMAX={field},SINUSOIDAL=1,INTEGRATION_ORDER={order},"
                    f"STEPS_PER_PERIOD={steps},SYNCH_RAD={sr},ISR=0\nBL:LINE=(W)\n"
                )
                out = track(lattice, p0, beam)
                # On y=py=0, magnetic impulses and scalar radiation losses are
                # independent of x and px. Antisymmetry cancels common impulses;
                # every collinear recoil must preserve the difference in Px/P.
                direction = out["xp"] / np.sqrt(1 + out["xp"] ** 2 + out["yp"] ** 2)
                ratio = float((direction[1] - direction[0]) / (2 * slope / np.hypot(1, slope)))
                close(out["p"][0], out["p"][1], 3e-11, "symmetric scalar loss")
                pout = float(out["p"][0])
                # Integrating the incorrect law dPx/Px=dP/P0 gives this ratio.
                # It diagnoses the discrepancy; direction conservation is the oracle.
                faulty = float(pin / pout * np.exp((pout - pin) / p0))
                if sr == 0:
                    close(ratio, 1.0, 4e-14, "radiation-off direction")
                    close(pout, pin, 3e-11, "radiation-off momentum")
                else:
                    if not pout < pin:
                        raise RuntimeError("Classical radiation did not reduce momentum")
                    if abs(ratio - 1) > 2e-12:
                        close(ratio, faulty, 2e-12, "continuum normalization diagnostic")
                        defect = True
                print(
                    f"serial/order={order} steps={steps} delta={delta:+.2f} SR={sr}: "
                    f"pout={pout:.17g} mc angular_response={ratio:.17g} expected=1 "
                    f"(collinear recoil); faulty_continuum={faulty:.17g}"
                )
    verdict(defect)


if __name__ == "__main__":
    main()
