"""COL-22: S-based cathode_fft_3d tracking silently applies no space charge

Selecting cathode_fft_3d with s-based tracking silently transports the bunch without space charge. This method is documented for time-based tracking only, so the incompatible configuration should be rejected. In a 1 cm drift, the exit position–momentum covariance is 1.000e−13 m and the longitudinal covariance is zero, exactly matching space charge off. The ordinary FFT method gives nonzero collective kicks for the same bunch.

The script tracks fresh identical Tao bunches with space charge off, ordinary FFT and cathode FFT
under s-based tracking. The ordinary FFT case controls that collective effects are active; the
cathode case must either report its unsupported configuration or reveal the silent space-charge-off
result.
"""

import numpy as np
from _bmad import compare, verdict
from _collective_fields import tao_covariance

off, rejected = tao_covariance("off")
if rejected:
    raise RuntimeError("Space-charge-off control rejected")
fft, rejected = tao_covariance("fft_3d")
if rejected or not np.isfinite(fft).all() or np.max(abs(fft - off)) < 1e-12:
    raise RuntimeError("Ordinary FFT collective-force control failed")
print(f"Space-charge-off sigma_12, sigma_56 (m): {off.tolist()}")
print(f"Ordinary FFT sigma_12, sigma_56 (m): {fft.tolist()}")
cathode, rejected = tao_covariance("cathode_fft_3d")
if rejected:
    print("S-based cathode configuration: rejected as required by the method contract")
    verdict(False)
else:
    # Rejection is the contract. Equality with SC off identifies the silent bypass.
    bypass = not compare(
        "s-based cathode sigma_12, sigma_56 (m), space-charge-off signature",
        cathode,
        off,
        atol=1e-25,
        rtol=2e-13,
    )
    if not bypass:
        raise RuntimeError("Accepted cathode configuration has a different tracking signature")
    print("S-based cathode configuration: observed=accepted without diagnostic; expected=rejected")
    verdict(True)
