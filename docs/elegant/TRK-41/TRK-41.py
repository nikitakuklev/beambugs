"""TRK-41: Standing-wave RFCA kick tracking fiducializes at the first kick, so PHASE=90 is not the crest

With STANDING_WAVE=1, BODY_FOCUS_MODEL=none, N_KICKS > 1 and automatic phasing, RFCA references
the first integration kick, so the documented PHASE=90 setting misses the crest. For a
half-wavelength gap (0.3747 m at 400 MHz, VOLT = 0.8000 MV) and 5.110 GeV electrons, the crest
is approximately 90°/N, where N is N_KICKS, with gain 512.6 keV for N = 8. PHASE=90 instead
gives approximately VOLT/N: 100.0 keV for N = 8 and 12.50 keV for N = 64. These formulas are
exact in the speed-of-light limit. The one-kick, matrix and travelling-wave controls keep the
crest at 90° to the tested ultrarelativistic tolerance.

The script sends four identical 5.110 GeV electrons through a half-wavelength RFCA gap of 0.8 MV at
400 MHz with STANDING_WAVE=1 and automatic phasing, for N_KICKS of 1, 2, 8 and 64, at PHASE = 90,
0 and 45. The gain is A sin(PHASE) + B cos(PHASE), so the first two runs give the crest phase and the
crest gain, which are compared with PHASE=90 and with the closed-form sum of the N kicks. The
travelling-wave model with 8 kicks and the standing-wave matrix method (N_KICKS = 0) are controls.
"""

import sys

import numpy as np
from _elegant import modes, particles, run, verdict

# SDDS constants.h, as Elegant uses them.
C, ME_MEV = 299792458.0, 0.51099906
FREQ, VOLT = 400e6, 800000.0  # Hz, V
LENGTH = C / (2 * FREQ)  # half a wavelength: the gap is crossed in half an RF period at beta = 1
P0 = 10000.0  # m_e c; beta = 1 - 5e-9 gives a relative fixed-speed flight-time correction of about 5e-9
COUNT = 4  # identical particles; Pelegant needs at least as many particles as ranks
# (label, STANDING_WAVE, N_KICKS); the last two are controls whose crest is PHASE=90
CASES = [
    ("standing wave, N_KICKS=1", 1, 1),
    ("standing wave, N_KICKS=2", 1, 2),
    ("standing wave, N_KICKS=8", 1, 8),
    ("standing wave, N_KICKS=64", 1, 64),
    ("standing wave, N_KICKS=0 (matrix)", 1, 0),
    ("travelling wave, N_KICKS=8", 0, 8),
]
PHASE_TOL = 1e-5  # degrees; the beta < 1 corrections are below 1e-6 degrees
GAIN_RTOL = 1e-7  # relative; the beta < 1 corrections are below 1e-8


def gain(mode, standing, kicks, phase):
    """Energy gain in eV of the beam (all particles identical) with automatic phasing."""
    lattice = (
        f"R: RFCA,L={LENGTH!r},VOLT={VOLT!r},FREQ={FREQ!r},PHASE={phase!r},"
        f"STANDING_WAVE={standing},N_KICKS={kicks},END1_FOCUS=0,END2_FOCUS=0\n"
        "BL: LINE=(R)\n"
    )
    commands = (
        f"&run_setup lattice=lattice.lte, use_beamline=BL, p_central={P0!r}, "
        "output=out.sdds, default_order=1 &end\n"
        "&run_control n_steps=1, n_passes=1 &end\n"
        "&sdds_beam input=beam.sdds &end\n"
        "&track soft_failure=0 &end\n"
    )
    beam = {"p": np.full(COUNT, P0)}
    out = particles(
        run({"lattice.lte": lattice, "run.ele": commands}, mode, {"beam.sdds": beam})["out.sdds"]
    )
    p = out["p"]
    if len(p) != COUNT or np.ptp(p) > 1e-12 * P0:
        sys.exit(f"{mode}: particles were lost or differ")
    return float((np.hypot(p[0], 1) - np.hypot(P0, 1)) * ME_MEV * 1e6)


def kick_sum(kicks, phase_deg):
    """Gain of the standing-wave kick model, fiducialized at its first kick, in the beta = 1 limit.

    Kick k (k = 0..N-1) sits at z_k = (k + 1/2) L/N and receives (V/N) sin(PHASE + k*delta), where
    delta = omega L/(N c) = pi/N for a half-wavelength gap: the field has no spatial phase, only
    the time phase advanced by the flight from the first kick. Summing the geometric series,
    G(phi) = (V/N) sin(phi + (N-1) delta/2) sin(N delta/2)/sin(delta/2). Its maximum, the crest,
    is (V/N)/sin(pi/(2N)) at phi = 90/N degrees; at PHASE = 90 the sum is exactly V/N.
    """
    n = max(kicks, 1)
    delta = np.pi / n
    phi = np.radians(phase_deg)
    return VOLT / n * np.sin(phi + (n - 1) * delta / 2) * np.sin(n * delta / 2) / np.sin(delta / 2)


defect = False
for mode in modes():
    for label, standing, kicks in CASES:
        g90, g0, g45 = (gain(mode, standing, kicks, phase) for phase in (90.0, 0.0, 45.0))
        # Gain linear in the voltage of each kick: G(phi) = A sin(phi) + B cos(phi) with A = G(90),
        # B = G(0); G(45) checks that form. The crest is atan2(A, B) and the crest gain hypot(A, B).
        if abs(g45 - (g90 + g0) / np.sqrt(2)) > GAIN_RTOL * VOLT:
            sys.exit(f"{mode} {label}: the gain is not sinusoidal in PHASE")
        crest = float(np.degrees(np.arctan2(g90, g0)))
        peak = float(np.hypot(g90, g0))
        name = f"{mode} {label}"
        if standing and kicks > 1:
            # Independent closed form of the same kicks: the crest gain and its phase.
            n = kicks
            expect_peak = float(VOLT / (n * np.sin(np.pi / (2 * n))))
            if abs(peak - expect_peak) > GAIN_RTOL * VOLT or abs(crest - 90 / n) > PHASE_TOL:
                sys.exit(f"{name}: the kick sum differs from its closed form")
            if abs(g90 - kick_sum(n, 90.0)) > GAIN_RTOL * VOLT:
                sys.exit(f"{name}: the PHASE=90 gain differs from V/N")
            print(
                f"{name}: crest at PHASE={crest:.9f} deg (closed form {90 / n:.9f}) with gain "
                f"{peak:.3f} eV (closed form {expect_peak:.3f}); PHASE=90: observed={g90:.3f} eV "
                f"expected={peak:.3f} eV (crest), ratio {g90 / peak:.4f}"
            )
            defect |= abs(crest - 90.0) > PHASE_TOL
        else:
            # One kick, the matrix method and the travelling wave: crest at 90 with gain V.
            print(
                f"{name}: crest at PHASE={crest:.6f} deg with gain {peak:.3f} eV; PHASE=90 gain {g90:.3f} eV"
            )
            if abs(crest - 90.0) > PHASE_TOL or abs(g90 - VOLT) > GAIN_RTOL * VOLT:
                sys.exit(f"{name}: control fails: PHASE=90 is not the crest with gain VOLT")
print(f"comparison: crest phase tolerance {PHASE_TOL:g} deg, gain tolerance {GAIN_RTOL:g} of VOLT")
verdict(defect)
