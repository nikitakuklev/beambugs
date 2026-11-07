"""IO-04: SDDS moment accumulation discards the earlier batches

The SDDS routines `accumulateMoments` and `accumulateWeightedMoments` return the mean, rms and standard deviation of data supplied in batches (first call with reset = 1, later calls with reset = 0), but in their default single-thread form they do not combine the batches. For the values 2, 4, 8, 10 supplied as [2, 4] then [8, 10], the mean is 6, the rms √46 = 6.782 and the sample standard deviation √(40/3) = 3.651, as one call over all four values gives. The unweighted routine returns a mean of 4.5 (the last batch's sum over the total count) and the unit-weight routine returns 9 (the last batch's mean).

Setup: `gcc`, and `sddsprocess` on `PATH` from an SDDS build tree, because the script compiles small C programs against that tree's `lib/Linux-x86_64/libmdbmth.a` and `libmdblib.a`.

The script compiles a small C program that calls the routines on the whole data and on the two batches, with unit weights and with weights 1, 2, 3, 4, and compares the combined mean, rms and standard deviation with closed forms; the one-call results and `computeMoments` are the controls.
"""

import sys

import numpy as np
from _elegant import verdict
from _sdds import c_check

RTOL = 1e-14
C_SOURCE = r"""
#include <stdio.h>
extern long computeMoments(double *, double *, double *, double *, double *, long);
extern long accumulateMoments(double *, double *, double *, double *, long, long);
extern long accumulateWeightedMoments(double *, double *, double *, double *, double *, long, long);

static void report(const char *name, double m, double r, double s) {
  printf("%s %.17g %.17g %.17g\n", name, m, r, s);
}

int main(void) {
  double x[] = {2, 4, 8, 10}, unit[] = {1, 1, 1, 1}, w[] = {1, 2, 3, 4};
  double mean, rms, sd, mad;
  if (!computeMoments(&mean, &rms, &sd, &mad, x, 4)) return 1;
  report("computeMoments", mean, rms, sd);
  if (!accumulateMoments(&mean, &rms, &sd, x, 4, 1)) return 2;
  report("unweighted-whole", mean, rms, sd);
  if (!accumulateMoments(&mean, &rms, &sd, x, 2, 1)) return 3;
  if (!accumulateMoments(&mean, &rms, &sd, x + 2, 2, 0)) return 4;
  report("unweighted-split", mean, rms, sd);
  if (!accumulateWeightedMoments(&mean, &rms, &sd, x, unit, 4, 1)) return 5;
  report("unit-weight-whole", mean, rms, sd);
  if (!accumulateWeightedMoments(&mean, &rms, &sd, x, unit, 2, 1)) return 6;
  if (!accumulateWeightedMoments(&mean, &rms, &sd, x + 2, unit + 2, 2, 0)) return 7;
  report("unit-weight-split", mean, rms, sd);
  if (!accumulateWeightedMoments(&mean, &rms, &sd, x, w, 4, 1)) return 8;
  report("weighted-whole", mean, rms, sd);
  if (!accumulateWeightedMoments(&mean, &rms, &sd, x, w, 2, 1)) return 9;
  if (!accumulateWeightedMoments(&mean, &rms, &sd, x + 2, w + 2, 2, 0)) return 10;
  report("weighted-split", mean, rms, sd);
  return 0;
}
"""


def fmt(values):
    return "[" + ", ".join(f"{v:.16g}" for v in values) + "] (mean, rms, sample sd)"


def moments(x, w):
    """Mean, rms and sample standard deviation of x with weights w, as the routines define them.

    mean = sum(w x)/sum(w), rms = sqrt(sum(w x^2)/sum(w)), and the standard deviation is
    sqrt((rms^2 - mean^2) n/(n - 1)) with n the number of values, not the sum of weights.
    """
    mean = np.sum(w * x) / np.sum(w)
    rms = np.sqrt(np.sum(w * x**2) / np.sum(w))
    return np.array([mean, rms, np.sqrt((rms**2 - mean**2) * len(x) / (len(x) - 1))])


x = np.array([2.0, 4.0, 8.0, 10.0])
weights = {"unit": np.ones(4), "weighted": np.array([1.0, 2.0, 3.0, 4.0])}
expected = {
    "computeMoments": moments(x, weights["unit"]),
    "unweighted-whole": moments(x, weights["unit"]),
    "unweighted-split": moments(x, weights["unit"]),
    "unit-weight-whole": moments(x, weights["unit"]),
    "unit-weight-split": moments(x, weights["unit"]),
    "weighted-whole": moments(x, weights["weighted"]),
    "weighted-split": moments(x, weights["weighted"]),
}
# Closed forms for the unit-weight data 2, 4, 8, 10: mean 6, rms sqrt(46), sd sqrt(40/3).
if not np.allclose(
    expected["computeMoments"], [6, np.sqrt(46), np.sqrt(40 / 3)], rtol=RTOL, atol=0
):
    sys.exit("the closed forms disagree with the moment definitions")
defect = False
for name, *words in c_check(C_SOURCE):
    observed = np.array(words, dtype=float)
    print(f"{name}: observed={fmt(observed)} expected={fmt(expected[name])}")
    agrees = np.allclose(observed, expected[name], rtol=RTOL, atol=0)
    if name.endswith("whole") or name == "computeMoments":
        if not agrees:
            sys.exit(f"{name}: a one-call control failed: the comparison is invalid")
    else:
        defect |= not agrees
print(f"comparison: rtol={RTOL:g}")
verdict(defect)
