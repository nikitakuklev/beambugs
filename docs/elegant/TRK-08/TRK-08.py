"""TRK-08: Finite matrix RFCA assigns the whole transit to the entrance speed

An RFCA of nonzero length using the default matrix method (N_KICKS = 0) times the entire cavity at
the entrance speed despite modeling acceleration in its transverse response. Without transverse
focusing, that response bounds the transit time for monotonic acceleration; kick tracking satisfies
its corresponding bounds. For a 0.75 m, 160 kV cavity at crest and electron momentum p = 2.3 mₑc,
the 2.728 ns transit exceeds the allowed upper limit by 27.45 ps. The minimum timing errors fall to
0.3778 ps at 10 mₑc and 0.3902 fs at 100 mₑc.

The script sends electrons of 2.3, 10 and 100 mₑc, on axis and with slopes of plus and minus 10⁻⁷,
through a 0.75 m RFCA of 160 kV (zero frequency or 1 MHz, crest phase, no end or body focusing,
default N_KICKS = 0). The slope pair measures the mean of 1/p along the cavity, and the convexity
of the flight time per length turns it into an interval for the transit time; the on-axis time is
compared with that interval. Controls are zero voltage, an exact drift, and kick tracking with one
and 128 kicks, which stay inside the interval; the uniform-field flight time is printed for
reference.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

# SDDS constants.h, as Elegant uses them.
C, ME_MEV = 299792458.0, 0.51099906
LENGTH, VOLT = 0.75, 160000.0  # m, V
EPS = 1e-7  # slope pair measuring the transverse response
MOMENTA = (2.3, 10.0, 100.0)  # entrance momentum, units of m_e c; the reference is the first
TOL = 3e-22  # s, agreement with the interval for the controls; a larger excess is the defect
# (label, length, voltage, FREQ, N_KICKS)
CASES = [
    ("matrix, FREQ=0", LENGTH, VOLT, 0.0, 0),
    ("matrix, FREQ=1 MHz", LENGTH, VOLT, 1e6, 0),
    ("matrix, zero voltage", LENGTH, 0.0, 0.0, 0),
    ("kicks, N_KICKS=1", LENGTH, VOLT, 0.0, 1),
    ("kicks, N_KICKS=128", LENGTH, VOLT, 0.0, 128),
]


def track(mode, length, volt, freq, kicks):
    """Exit x, t, p of the nine particles: per momentum, slopes 0, +EPS and -EPS, all at x = 0."""
    lattice = (
        f"R: RFCA,L={length!r},VOLT={volt!r},PHASE=90,FREQ={freq!r},N_KICKS={kicks},"
        "END1_FOCUS=0,END2_FOCUS=0,CHANGE_T=0,CHANGE_P0=0,T_REFERENCE=0\n"
        "BL: LINE=(R)\n"
    )
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={MOMENTA[0]!r}, "
        "output=out.sdds, default_order=2 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track soft_failure=0 &end\n"
    )
    beam = {"p": np.repeat(MOMENTA, 3), "xp": np.tile([0.0, EPS, -EPS], len(MOMENTA))}
    out = particles(
        run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
    )
    if len(out["p"]) != len(beam["p"]):
        sys.exit(f"{mode}: particles were lost")
    return out


def allowed_interval(p_in, p_out, rbar, length):
    """Transit time bounds from the mean of r = 1/p along the cavity, for any profile p(z).

    c dt/dz = gamma/p = f(r) with f(r) = sqrt(1 + r^2), convex. For a monotonic p(z), r(z) lies
    between r_out = 1/p_out and r_in = 1/p_in. Jensen's inequality gives the mean of f(r) >=
    f(rbar); the chord of f through the end values lies above f and is linear in r, so the mean of
    f(r) <= chord(rbar). The time is L times the mean of f divided by c.
    """
    r_in, r_out = 1 / p_in, 1 / p_out
    weight = (rbar - r_out) / (r_in - r_out)
    if not -1e-9 < weight < 1 + 1e-9:
        sys.exit("the measured mean of 1/p lies outside the end values")
    f_in, f_out = np.sqrt(1 + r_in**2), np.sqrt(1 + r_out**2)
    lower = length / C * np.sqrt(1 + rbar**2)
    upper = length / C * (weight * f_in + (1 - weight) * f_out)
    return float(lower), float(upper)


defect = False
for mode in modes():
    for label, length, volt, freq, kicks in CASES:
        out = track(mode, length, volt, freq, kicks)
        for index, p_in in enumerate(MOMENTA):
            on_axis, plus, minus = 3 * index, 3 * index + 1, 3 * index + 2
            p_out, observed = float(out["p"][on_axis]), float(out["t"][on_axis])
            if not np.isfinite([p_out, observed]).all():
                sys.exit(f"{mode} {label}, p_in={p_in:g}: nonfinite time or momentum")
            gamma_in, gamma_out = np.hypot(p_in, 1), np.hypot(p_out, 1)
            gain = volt / (1e6 * ME_MEV)  # energy gain in units of m_e c^2
            if not freq and abs(gamma_out - gamma_in - gain) > 7e-13:
                sys.exit(f"{mode} {label}: the energy gain is not V/(m c^2/e)")
            name = f"{mode} {label}, p_in={p_in:g}"
            if not volt:
                # No field: an exact drift, t = L gamma/(p c).
                expected = float(length * gamma_in / (p_in * C))
                print(f"{name}: observed=[{observed!r}] expected=[{expected!r}]")
                if abs(observed - expected) > TOL:
                    sys.exit(f"{name}: the zero-voltage control is not an exact drift")
                continue
            # Without focusing p_x is conserved, so to first order x = p_x * integral dz/p(z).
            # With p_x = p_in*EPS/sqrt(1 + EPS^2) (units of m_e c), the slope pair gives
            # rbar = (x_plus - x_minus)/(2 p_x L), the mean of 1/p over the cavity.
            px = p_in * EPS / np.sqrt(1 + EPS**2)
            rbar = float(out["x"][plus] - out["x"][minus]) / (2 * px * length)
            lower, upper = allowed_interval(p_in, p_out, rbar, length)
            if not all(np.isfinite([observed, rbar, lower, upper])):
                sys.exit(f"{mode} {label}, p_in={p_in:g}: nonfinite time or interval endpoint")
            late = max(observed - upper, lower - observed, 0.0)
            note = f"outside by {late:.4g} s" if late > TOL else "inside"
            if freq == 0.0 and gain:
                # Uniform field: dgamma/dz = gain/L, so dt/dz = gamma/(c p) integrates to
                # L (p_out - p_in)/(c gain).
                uniform = float(length * (p_out - p_in) / (gain * C))
                note += f"; uniform-field time {uniform!r}"
            print(f"{name}: observed=[{observed!r}] expected=[{lower!r}, {upper!r}] {note}")
            if kicks:
                if late > TOL:
                    sys.exit(f"{name}: the kick-tracking control violates the interval")
            else:
                defect |= late > TOL
print(f"comparison: interval tolerance {TOL:g} s")
verdict(defect)
