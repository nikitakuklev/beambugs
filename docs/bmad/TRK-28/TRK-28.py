"""TRK-28: Magnus solenoid spin omits the tracked/reference charge ratio

Forward Magnus spin tracking through a solenoid gives equal-momentum electrons and positrons the same rotation in the same physical field. The normalized solenoid strength needs the tracked-to-reference charge ratio in the Thomas–BMT generator. For a positron at 3 MeV/c in an electron-reference 1.3 m solenoid of strength −1.1 m⁻¹, spin (0.36, 0.48, 0.8) becomes (−0.4254, 0.4231, 0.8) instead of (0.5253, −0.2899, 0.8). Matching-charge cases and the two ordinary integration methods agree with the charge-aware expectation.

The script sweeps equal-mass electron and positron spins in both reference-charge lattices using BMAD_DIST. It compares forward Magnus transport with the charge-aware axial Thomas–BMT rotation and validates ordinary integration, Runge–Kutta, matching-charge, zero-field and longitudinal-spin controls.
"""

import numpy as np
from _bmad import compare, verdict
from _rf_and_spin import control, magnus_records, rotate

records = magnus_records()
same_error = opposite_error = 0.0
for ident, same_charge, theta, seed, forward, returned, backward in records:
    if ident[-1] != 1:
        continue
    error = float(np.max(np.abs(forward - rotate(seed, theta))))
    if same_charge:
        same_error = max(same_error, error)
    else:
        opposite_error = max(opposite_error, error)
    if ident == (1, 2, 2, 2, 1, 1):
        print(
            f"3 MeV/c positron: forward={forward.tolist()} expected={rotate(seed, theta).tolist()}"
        )
control("same-charge forward Magnus T-BMT maximum residual", same_error, 0, atol=3e-15)
verdict(
    compare("opposite-charge forward Magnus T-BMT maximum residual", opposite_error, 0, atol=3e-15)
)
