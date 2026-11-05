"""XLT-02: Converter momentum cuts fail to restrict generated momenta

A converter with output momentum limits still generates particles outside the requested interval. With a 200–1000 keV/c table and limits of 350–780 keV/c, 65 of 128 fixed-seed particles fall outside the limits, spanning 202.5–999.8 keV/c. The integrated weight reflects the cut while inverse sampling uses the original table-cell endpoints. Clipping the cumulative distribution and its inversion to the same endpoints is required to preserve the requested momentum support.

The script samples the supplied momentum distribution with and without the output limits. It counts particles outside the permitted interval and compares the cut weight with the uniform-density acceptance; the random seed is 46731.
"""

import numpy as np
from _bmad import compare, verdict
from _s3_24_converter import samples

base, _, _ = samples()
if compare("unrestricted weight, unit-integral table", base[0, 1], 1, atol=2e-14):
    raise RuntimeError("Unrestricted normalization control failed")
if np.any((base[:, 0] < 200000) | (base[:, 0] > 1000000)):
    raise RuntimeError("Unrestricted samples left the supplied grid")
cut, _, _ = samples("cv[pc_out_min]=350000\ncv[pc_out_max]=780000\n")
outside = int(np.count_nonzero((cut[:, 0] < 350000) | (cut[:, 0] > 780000)))
print(
    f"cut momentum range (eV/c): observed={[float(cut[:, 0].min()), float(cut[:, 0].max())]} permitted=[350000,780000]"
)
# Uniform density in p: accepted fraction is interval length divided by full width.
# A sharp integration boundary gives a small quadrature residual, separately from support.
compare("cut yield, uniform momentum density", cut[0, 1], 430000 / 800000, atol=1e-4)
verdict(compare("samples outside momentum limits, out of 128", outside, 0, atol=0))
