"""XLT-03: Converter beta changes rescale longitudinal position backwards

Converting a particle to a different speed rescales its longitudinal coordinate z by the inverse of the required speed ratio. The coordinate definition z = −βc(t − t₀), where β is speed divided by c and t₀ is reference time, requires z to scale as β_new/β_old when stored time stays fixed. In a 4 MeV/c electron-to-positron example, the returned z is 31.63 mm instead of 16.72 mm. The absolute clock and the post-drift speed-ratio calculation give the same expected value.

The script compares the first fixed-seed conversion with a drift-only clock control. It checks the unchanged time and computes z both from the absolute clock and from the required speed ratio.
"""

from _bmad import compare, verdict
from _s3_24_converter import samples

_, clock, absolute = samples()
oldbeta, newbeta, oldtime, newtime, oldz, actual, expected, _ = clock
if compare("stored time through conversion (s)", newtime, oldtime, atol=2e-24):
    raise RuntimeError("Conversion changed the clock")
# z=-beta*c*(t-t0); changing beta at fixed time scales z by beta_new/beta_old.
if compare(
    "absolute clock versus speed-ratio oracle (m)",
    absolute[1],
    oldz * newbeta / oldbeta,
    atol=2e-17,
):
    raise RuntimeError("Independent clock controls disagree")
verdict(compare("converted z (m), fixed absolute time", actual, expected, atol=2e-17))
