"""COL-25: MPI TRWAKE adds fixed-bin histograms whose time origins differ between ranks

With multiple bunches and a fixed TRWAKE bin count (N_BINS > 0) under MPI (Pelegant), each tracking
rank centers its histogram on its own bunch mean, so unequal rank means make summed bins represent
different physical times. The correct kick is a causal sum within each bunch; serial tracking, MPI
with equal rank means and automatic bins reproduce it. For two 80.00 pC bunches whose early and late
halves occupy different ranks, the first particle of a late half receives no horizontal slope kick
instead of 6.720×10⁻⁷, and the last particle's kick is 60.00% too small.

The script tracks two 24-particle bunches, 3 ns apart, through one dipole TRWAKE with a linear
causal wake and 64 fixed bins. Every particle has the same transverse offsets, so the correct kick
is a direct causal pair sum over the particles of the same bunch, written without any binning. In
one input row order both tracking ranks see the same time distribution; in the other, one rank holds
the early half and the other the late half of each bunch, so their mean times differ. A prediction
that replaces each time by its offset from its rank's mean explains the result; serial runs, equal
rank means and automatic bins are passing controls, while halving the bin width leaves the
erroneous result unchanged.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

REST_EV = 510999.06  # SDDS constants.h: electron rest energy, eV
P = 41.0  # electron momentum, units of m_e c
SLOPE = 4e25  # V/C/m per s; Wx(tau) = SLOPE tau for tau >= 0, Wy = 0.7 Wx, W(0) = 0
CHARGE = 80e-12  # C per bunch
X0, Y0 = 1.1e-3, -0.6e-3  # m, the same for every particle
# Offsets from the bunch center (ps), four particles at each; bunch centers 2 ns and 5 ns.
OFFSETS = np.repeat([-10, -6, -2, 2, 6, 10], 4) * 1e-12
N = len(OFFSETS)  # particles per bunch
CENTERS = (2e-9, 5e-9)
BINS = 64
RTOL, ATOL = 2e-12, 1e-19


def wake_table(spacing):
    """ASCII SDDS table of Wx, Wy on 257 points of the given spacing."""
    rows = "".join(
        f"{tau!r} {SLOPE * tau!r} {0.7 * SLOPE * tau!r}\n"
        for tau in (float(n) * spacing for n in range(257))
    )
    return (
        "SDDS1\n"
        "&column name=time, units=s, type=double &end\n"
        "&column name=wx, units=V/C/m, type=double &end\n"
        "&column name=wy, units=V/C/m, type=double &end\n"
        "&data mode=ascii, no_row_counts=1 &end\n" + rows
    )


def kicks(mode, layout, spacing, bins):
    """Slope kicks (xp, yp) of the 2N particles, ordered by particle ID."""
    t = np.concatenate([OFFSETS + c for c in CENTERS])
    ids = np.arange(1, 2 * N + 1)
    if layout == "skew":
        # One rank gets the early half of each bunch, the other the late half.
        order = np.r_[0:12, 24:36, 12:24, 36:48]
    else:
        # Both ranks get two of the four particles at each time of each bunch.
        half = np.arange(N).reshape(6, 4)[:, :2].ravel()
        other = np.arange(N).reshape(6, 4)[:, 2:].ravel()
        order = np.r_[half, half + N, other, other + N]
    beam = {
        "x": np.full(2 * N, X0)[order],
        "y": np.full(2 * N, Y0)[order],
        "t": t[order],
        "p": np.full(2 * N, P),
        "particleID": ids[order],
        "IDSlotsPerBunch": N,
    }
    lattice = (
        f"Q: CHARGE,TOTAL={2 * CHARGE!r}\n"
        'W: TRWAKE,INPUTFILE="wake.sdds",TCOLUMN="time",WXCOLUMN="wx",WYCOLUMN="wy",'
        f"N_BINS={bins},BUNCHED_BEAM_MODE=1,SMOOTHING=0,INTERPOLATE=0,"
        "X_DRIVE_EXPONENT=1,Y_DRIVE_EXPONENT=1,X_PROBE_EXPONENT=0,Y_PROBE_EXPONENT=0\n"
        "BL: LINE=(Q,W)\n"
    )
    commands = (
        "&global_settings echo_namelists=0 &end\n"
        f"&run_setup lattice=lattice.lte,use_beamline=BL,p_central={P!r},output=out.sdds &end\n"
        "&run_control n_steps=1,n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds,use_bunched_mode=1,track_pages_separately=0 &end\n"
        "&track &end\n"
    )
    result = run(
        {"lattice.lte": lattice, "run.ele": commands, "wake.sdds": wake_table(spacing)},
        mode,
        {"beam.sdds": beam},
    )
    out = particles(result["out.sdds"])
    if not np.array_equal(out["particleID"], ids):
        sys.exit(f"{mode}: particles lost")
    # The wake changes only the slopes, which start at zero.
    return np.column_stack([out["xp"], out["yp"]])


def pair_sum(times):
    """Slope kicks (xp, yp) of one bunch from a direct causal pair sum over its particles.

    Particle i receives dq'_i = (Q/N) sum_j q_j W_q(t_i - t_j)/(E0 p) for q = x, y, summed over the
    particles j of the same bunch behind which it arrives (W(0) = 0), with Q/N the macroparticle
    charge, q_j the source's offset and E0 the electron rest energy in eV. The times lie on the
    wake table's grid, so no interpolation enters.
    """
    delay = np.maximum(times[:, None] - times[None, :], 0.0)
    voltage = (CHARGE / N) * SLOPE * delay.sum(axis=1)  # V/m, times the source offset
    return np.column_stack([X0 * voltage, 0.7 * Y0 * voltage]) / (REST_EV * P)


def fmt(values):
    return "[" + ", ".join(f"{v:.16g}" for v in np.ravel(values) + 0.0) + "]"


def compare(label, observed, expected):
    """Print the kicks of particles 13 (first of the late half) and 24 of the first bunch."""
    for index in (12, 23):
        print(
            f"{label}, particle {index + 1}: "
            f"observed={fmt(observed[index])} expected={fmt(expected[index])}"
        )
    return bool(np.allclose(observed, expected, rtol=RTOL, atol=ATOL))


defect = False
for mode in ("serial", *modes(mpi_only=True)):
    for layout, spacing, bins in (
        ("balanced", 2e-12, BINS),
        ("skew", 2e-12, BINS),
        ("skew", 1e-12, BINS),
        ("skew", 2e-12, 0),
    ):
        expected = np.vstack([pair_sum(OFFSETS)] * 2)
        auto = "automatic bins" if bins == 0 else f"{bins} bins of {spacing * 1e12:g} ps"
        label = f"{mode} {layout} rows, {auto}"
        observed = kicks(mode, layout, spacing, bins)
        agrees = compare(label, observed, expected)
        if not (mode == "mpi" and layout == "skew" and bins):
            if not agrees:
                sys.exit(f"{label}: control disagrees with the pair sum; the comparison is invalid")
            continue
        # Cause: the early-half rank centers its grid 6 ps before the bunch center and the
        # late-half rank 6 ps after it, so the summed histogram pairs times that differ by 12 ps.
        # Each particle then behaves as if it arrived at its offset from its own rank's mean time.
        shifted = OFFSETS - np.where(np.arange(N) < N // 2, -6e-12, 6e-12)
        local = np.vstack([pair_sum(shifted)] * 2)
        if not np.allclose(observed, local, rtol=RTOL, atol=ATOL) and not agrees:
            sys.exit(f"{label}: kicks follow neither the pair sum nor the rank-local origins")
        print(f"{label}, particle 13: rank-local-origin prediction={fmt(local[12])}")
        print(f"{label}: particle 24 kick is {1 - observed[23, 0] / expected[23, 0]:.4%} too small")
        defect |= not agrees
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g}")
verdict(defect)
