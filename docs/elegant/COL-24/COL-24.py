"""COL-24: Single-bunch TRWAKE fixed bins retain a zero time reference

With a fixed TRWAKE bin count (N_BINS > 0) and a single bunch, Elegant anchors the grid to time zero
rather than the bunch and omits particles outside it as both wake sources and recipients. A
stationary wake depends on arrival-time differences, and automatic-bin controls recover the full
causal pair-sum kicks. For a 0.6000 nC electron bunch represented by 18 macroparticles spanning
52.00 ps, 96 bins of 2.000 ps give a trailing horizontal slope kick of 1.302×10⁻⁵ at zero shift but
zero after +0.7000 ns or a 0.2100 m drift, with an unbinned-particle warning. A +80.00 ps shift
leaves only 12 particles in the grid.

The script tracks the same bunch through TRWAKE with a damped-sine wake after different arrival-time
shifts and a drift, with 96 fixed bins, 1024 fixed bins and automatic bins, and with the bunch and
wake table stretched to 4 ps. It compares every slope with a causal pair sum over the particles,
which is shift-invariant, and for the cases with particles outside the grid also with the pair sum
restricted to the particles inside it. The zero shift, +28.00 ps (inside the grid), 1024-bin and
automatic-bin cases are the controls.
"""

import re
import sys

import numpy as np
from _elegant import modes, verdict
from _trwake import ATOL, RTOL, N, beam, compare, fmt, pair_sum, track

C = 299792458.0  # SDDS constants.h
P = 37.0  # m_e c
NBINS = 96
# Six time groups of three particles, centered on the bin centers of a grid of spacing `scale`*2 ps.
GROUPS = np.array([-26, -16, -6, 6, 16, 26]) * 1e-12

# (label, bins, arrival-time shift in s, drift in m, scale of times and wake spacing)
CASES = (
    ("96 bins, no shift", NBINS, 0.0, 0.0, 1),
    ("96 bins, +28 ps", NBINS, 28e-12, 0.0, 1),
    ("96 bins, +80 ps", NBINS, 80e-12, 0.0, 1),
    ("96 bins, +0.7 ns", NBINS, 0.7e-9, 0.0, 1),
    ("96 bins, 0.21 m drift", NBINS, 0.0, 0.21, 1),
    ("1024 bins, +0.7 ns", 1024, 0.7e-9, 0.0, 1),
    ("automatic bins, +0.7 ns", 0, 0.7e-9, 0.0, 1),
    ("96 bins, 4 ps, +1.4 ns", NBINS, 1.4e-9, 0.0, 2),
    ("automatic bins, 4 ps, +1.4 ns", 0, 1.4e-9, 0.0, 2),
)

defect = False
for mode in modes():
    for label, bins, shift, drift, scale in CASES:
        spacing = scale * 2e-12
        times = np.repeat(GROUPS * scale, 3) + shift
        columns = beam(times, 0.6e-3, -0.4e-3, P)
        observed, log = track(mode, columns, n_bins=bins, drift=drift, spacing=spacing)
        # A pair sum depends on arrival-time differences only, so the expected kicks do not depend
        # on the shift or the drift.
        expected = pair_sum(columns, spacing=spacing)
        # Particles Elegant bins: with the grid centered on time zero, tmin = -bins spacing/2 and
        # bin k is centered on tmin + k spacing. The index is (t - tmin)/spacing + 0.5 truncated
        # toward zero and accepted for 0 <= k <= bins-1, i.e. tmin - 1.5 spacing < t <
        # tmin + (bins - 0.5) spacing. A drift adds drift/(beta c) to the arrival time.
        arrival = times + drift * np.sqrt(1 + P**2) / (C * P)
        tmin = -bins * spacing / 2
        inside = np.ones(N, dtype=bool)
        if bins > 0:
            inside = (arrival > tmin - 1.5 * spacing) & (arrival < tmin + (bins - 0.5) * spacing)
        warned = "not binned in TRWAKE" in log
        if inside.all():
            bad = compare(f"{mode} {label}", observed, expected)
            if bad or warned:
                sys.exit(f"{mode} {label}: control disagrees with the pair sum")
            continue
        # Cause: particles outside the grid are skipped as sources and as probes.
        predicted = pair_sum(columns, spacing=spacing, binned=inside)
        bad = compare(f"{mode} {label}, {inside.sum()} of {N} in the grid", observed, expected)
        if inside.any():
            middle = np.flatnonzero(inside)[-1]
            print(
                f"{mode} {label}: last particle in the grid (t={arrival[middle] * 1e12:g} ps): "
                f"observed={fmt(observed[middle])} expected={fmt(expected[middle])}"
            )
        if bad:
            # A mismatch must be explained by the restricted pair sum and the warning to count.
            if not np.allclose(observed, predicted, rtol=RTOL, atol=ATOL):
                sys.exit(f"{mode} {label}: kicks differ from the pair sum restricted to the grid")
            if not warned:
                sys.exit(f"{mode} {label}: no unbinned-particle warning")
            # The serial warning states the count of binned particles.
            count = re.search(r"Only (\d+) of \d+ particles were binned", log)
            if mode == "serial" and (count is None or int(count[1]) != inside.sum()):
                sys.exit(f"{mode} {label}: Elegant bins a different number of particles")
            defect = True
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} on the slopes of all {N} particles")
verdict(defect)
