"""TRK-27: Magnus solenoid spin fails backward-time inversion

Backward-time Magnus spin tracking through an axial solenoid repeats the forward rotation instead of undoing it. The constant-field spin generator requires opposite rotation angles for the two time directions. At 3 MeV/c, a 1.3 m solenoid with strength −1.1 m⁻¹ maps spin (0.36, 0.48, 0.8) to (−0.4780, −0.3626, 0.8) after a round trip instead of restoring it. Ordinary spin integration and Runge–Kutta tracking restore the spin to roundoff.

The script sweeps both reference charges, two momenta, three solenoids and three momentum offsets using BMAD_DIST. Matching-charge Magnus round trips and fresh backward passes are compared with Cartesian Thomas–BMT rotations, with ordinary integration, Runge–Kutta, zero-field and longitudinal-spin controls.
"""

import numpy as np
from _bmad import compare, verdict
from _rf_and_spin import control, magnus_records, rotate

records = magnus_records()
forward_error = return_error = backward_error = 0.0
for ident, same_charge, theta, seed, forward, returned, backward in records:
    config, ie, isp, idelta, ispin, method = ident
    if method != 1 or not same_charge:
        continue
    forward_error = max(forward_error, float(np.max(np.abs(forward - rotate(seed, theta)))))
    return_error = max(return_error, float(np.max(np.abs(returned - seed))))
    backward_error = max(backward_error, float(np.max(np.abs(backward - rotate(seed, -theta)))))
    if ident == (1, 2, 1, 2, 1, 1):
        print(
            f"3 MeV/c example: forward angle={theta:.16g} rad; round trip={returned.tolist()} expected={seed.tolist()}"
        )
control("same-charge forward Magnus T-BMT maximum residual", forward_error, 0, atol=3e-15)
a = compare("same-charge Magnus round-trip maximum residual", return_error, 0, atol=3e-15)
b = compare(
    "same-charge Magnus fresh backward T-BMT maximum residual", backward_error, 0, atol=3e-15
)
verdict(a or b)
