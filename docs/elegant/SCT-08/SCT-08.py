"""SCT-08: Smooth IBS shifts the beam centroid when there is dispersion and a momentum offset

With the smooth method (SMOOTH=1), IBSCATTER measures the beam centroid c and scales the transverse
coordinates about c + ημ, where η is the dispersion and μ the beam's mean fractional momentum offset
δ. The centroid already contains the dispersive displacement ημ, so the origin is displaced twice
and the centroid moves by (1−a)ημ, where a is the amplitude growth factor, instead of staying fixed.
In a closed ring with η = 14.29 m and μ = 1×10⁻⁴, an amplitude growth of 0.8501% moves the mean
horizontal position by −12.14 µm instead of 0; with μ = 0 it stays put. Users of smooth IBS with
dispersion and an off-momentum beam are affected.

The script tracks a 256-particle beam once around a ring of sixteen identical sector bends, with
horizontal dispersion η = 14.29 m, ending in an IBSCATTER with the smooth method. It compares the
mean horizontal position with that of a run with FACTOR=0, once with a mean momentum offset μ =
1×10⁻⁴ and once with μ = 0 as the control. Centered emittance growth leaves the centroid where it
is, so the expected shift is zero; the script also evaluates the shift (1−a)ημ that the displaced
scaling origin predicts.
"""

import sys

import numpy as np
from _elegant import modes, verdict
from _ibs_ring import ETA_X, FACTOR, P0, beam, track

ATOL = 1e-15  # m; positions are about 1.4e-3 m, so rounding is about 3e-19 m
FORMULA_RTOL, FORMULA_ATOL = 1e-9, 2e-18  # shift against the displaced-origin prediction


def fmt(values):
    return "[" + ", ".join(f"{v:.12g}" for v in np.ravel(values)) + "]"


def compare(label, observed, expected, rtol=0.0, atol=ATOL):
    bad = not np.allclose(observed, expected, rtol=rtol, atol=atol)
    print(f"{label}: observed={fmt(observed)} expected={fmt(expected)}")
    return bad


defect = False
print("mean x shift in m caused by the IBS update (FACTOR minus FACTOR=0)")
for mode in modes():
    for label, mean_delta in (("mean delta 0 control", 0.0), ("mean delta 1e-4", 1e-4)):
        columns = beam(mean_delta)
        reference = track(mode, columns, 0.0)
        out = track(mode, columns, FACTOR)
        shift = np.mean(out["x"]) - np.mean(reference["x"])
        # The mean momentum offset after the update must be unchanged, or the shift is not the
        # dispersive displacement of this offset.
        mu = np.mean(out["p"]) / P0 - 1
        if abs(mu - (np.mean(reference["p"]) / P0 - 1)) > 1e-12:
            sys.exit(f"{mode}: the update changed the mean momentum offset")
        # Coordinates scale by a about the origin c + eta mu, where c is the measured centroid,
        # so the centroid moves by (1 - a) eta mu, with a**2 the variance ratio of x.
        a = np.sqrt(np.var(out["x"]) / np.var(reference["x"]))
        bad = compare(f"{mode} {label}", shift, 0.0)
        if compare(
            f"{mode} {label}, displaced-origin prediction",
            shift,
            (1 - a) * ETA_X * mu + 0.0,
            FORMULA_RTOL,
            FORMULA_ATOL,
        ):
            sys.exit(f"{mode}: the shift does not follow (1-a)*eta*mu")
        if mean_delta == 0.0 and bad:
            sys.exit(f"{mode}: the control moved the centroid: the comparison is not valid")
        defect |= bad
print(f"comparison: atol={ATOL:g} m")
verdict(defect)
