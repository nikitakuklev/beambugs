"""COL-21: Transverse slice space charge inherits the last shielding-image factor

Combining transverse slice space charge with shielding images multiplies the kick by −2 for an odd image count and by +2 for an even nonzero count. Shielding images belong to the coherent synchrotron radiation calculation and should not change the slice-model force. In a 1 cm drift with 10 MeV/c electrons and 10 pC bunch charge, the space-charge change in the position–momentum covariance is 3.231e−11, −6.462e−11 and 6.462e−11 m for zero, one and two images. Longitudinal slice space charge is calculated at the direct-source iteration and avoids this multiplier.

The script runs fresh deterministic Tao bunches with space charge off and with slice space charge at
zero, one and two shielding images. It subtracts the drift-only covariance and requires all 2000
particles to survive. With CSR off, the tracked covariance must be independent of image count.
"""

import numpy as np
from _bmad import compare, verdict
from _collective_fields import tao_covariance

off, rejected = tao_covariance("off")
if rejected:
    raise RuntimeError("Space-charge-off control rejected")
changes = []
for images in (0, 1, 2):
    result, rejected = tao_covariance("slice", images)
    if rejected or not np.isfinite(result).all():
        raise RuntimeError("Slice tracking failed")
    changes.append(result[0] - off[0])
    print(
        f"images={images}: exit sigma_12, sigma_56 (m)={result.tolist()}; SC change sigma_12={changes[-1]:.16g}"
    )
if changes[0] <= 1e-12:
    raise RuntimeError("Unshielded transverse-force control failed")
# With CSR off, shielding images must not affect the slice-model force or the
# evolved bunch. This end-to-end test executes the image loop itself.
ratios = np.array(changes[1:]) / changes[0]
defect = compare(
    "tracked slice covariance ratios for images 1,2; image-independent force",
    ratios,
    np.ones(2),
    atol=0,
    rtol=2e-10,
)
verdict(defect)
