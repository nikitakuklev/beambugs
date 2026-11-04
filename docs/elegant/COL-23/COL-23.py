"""COL-23: TRWAKE quadrupole probe coordinates omit element misalignment

TRWAKE quadrupole wakes (probe exponents 1, drive exponents 0) use each particle's global transverse
coordinate even when the structure has misalignments DX and DY. The kick must use displacement from
the structure, so translating beam and structure together must preserve the kicks, and a beam on the
structure's axis must receive none. For a 0.6000 nC electron bunch represented by 18 macroparticles,
a common shift of (3.100, −1.700) mm raises the trailing particle's horizontal slope kick from
1.498×10⁻⁵ to 6.660×10⁻⁵, and an on-axis beam receives 5.161×10⁻⁵ instead of zero. Dipole wakes,
with probe factor 1, pass the translation controls.

The script tracks the same 18 electron macroparticles through a TRWAKE element with a damped-sine
wake on a 2 ps grid, for the quadrupole wake centered and translated, a beam on the translated
axis, and the dipole wake translated as the control. It compares every slope with a direct causal
pair sum over the structure-relative coordinates, and shows that evaluating the probe at the global
coordinate reproduces the observed slopes.
"""

import sys

import numpy as np
from _elegant import modes, verdict
from _trwake import ATOL, RTOL, N, beam, compare, pair_sum, track

# Six arrival-time groups of three particles; offsets and momenta vary across the bunch.
TIMES = np.repeat(np.arange(-25, 26, 10), 3) * 1e-12  # s
RELATIVE = np.column_stack([np.linspace(-0.8, 0.9, N), np.linspace(0.5, -0.7, N)]) * 1e-3  # m
MOMENTUM = np.linspace(31, 49, N)  # m_e c
SHIFT = (3.1e-3, -1.7e-3)  # m, common translation of beam and structure

# (label, shift, quadrupole wake, beam on the structure axis)
CASES = (
    ("quadrupole, centered", (0.0, 0.0), True, False),
    ("quadrupole, translated (3.1, -1.7) mm", SHIFT, True, False),
    ("quadrupole, translated (-4.3, 2.6) mm", (-4.3e-3, 2.6e-3), True, False),
    ("quadrupole, beam on the translated axis", SHIFT, True, True),
    ("dipole control, translated (3.1, -1.7) mm", SHIFT, False, False),
)

defect = False
for mode in modes():
    for label, shift, quadrupole, on_axis in CASES:
        position = np.broadcast_to(shift, (N, 2)).copy()
        if not on_axis:
            position += RELATIVE
        columns = beam(TIMES, position[:, 0], position[:, 1], MOMENTUM)
        drive = 0 if quadrupole else 1
        observed, _ = track(mode, columns, shift=shift, drive=drive)
        # Expected: the pair sum over coordinates relative to the structure.
        expected = pair_sum(columns, shift=shift, drive=drive)
        # Cause: the probe factor taken in the global frame instead.
        global_probe = pair_sum(columns, shift=shift, drive=drive, probe_offset=(0.0, 0.0))
        bad = compare(f"{mode} {label}", observed, expected)
        translated = shift != (0.0, 0.0)
        if translated and quadrupole:
            if bad:
                # A mismatch must be explained by the global-frame probe to count as the defect.
                if not np.allclose(observed, global_probe, rtol=RTOL, atol=ATOL):
                    sys.exit(f"{mode} {label}: kick follows neither the pair sum nor global probe")
            else:
                print(f"{mode} {label}: kick agrees with the structure-relative pair sum")
            defect |= bad
        elif bad:
            sys.exit(
                f"{mode} {label}: control disagrees with the pair sum; the comparison is invalid"
            )
print(f"comparison: rtol={RTOL:g}, atol={ATOL:g} on the slopes of all {N} particles")
verdict(defect)
