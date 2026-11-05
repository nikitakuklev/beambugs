"""TRK-29: SOLE tracks with B when KS is also set, while its matrix and Twiss output use KS

A SOLE solenoid given both a nonzero strength KS and a nonzero field B is tracked with the strength
that B implies, while its matrix and Twiss output use KS; the parameter description says B is used
only if KS is zero. For an electron at momentum 200 mₑc in a 0.65 m solenoid with KS = 0.9000 m⁻¹
and B = −0.02500 T (equivalent to KS = 0.07333 m⁻¹), the matrix output gives the documented
transfer element R₁₁ = cos²(KS·L/2) = 0.9169, with L the length, but tracked particles follow R₁₁ =
0.9994. With spin tracking, an on-axis spin turns by 0.04772 rad instead of 0.5857 rad. Either
parameter alone gives consistent tracking and optics.

The script runs a 0.65 m SOLE with KS alone, B alone and both, and for each compares the tracked
R₁₁ (x_out/x_in of one particle), the R₁₁ of matrix_output and the spin of an on-axis particle with
the analytic solenoid matrix and Thomas–BMT precession for the strength the parameter description
selects. The KS-only and B-only runs are the controls; with both set, the matrix output follows KS
and tracking follows B.
"""

import sys

import numpy as np
from _elegant import modes, verdict
from _sole import A_ELECTRON, ME, QE, C, axial_rotation, compare, on_axis_angle, track

LENGTH, P0 = 0.65, 200.0
KS, FIELD = 0.9, -0.025  # m^-1 and T
X0 = 1e-6  # m, initial x of the particle that measures R11
ATOL = 1e-13
MATRIX = '&matrix_output SDDS_output="matrix.sdds", SDDS_output_order=1 &end\n'


def r11(ks):
    """R11 = cos^2(KS L/2) of the solenoid transfer matrix (rotation by KS L/2 times focusing)."""
    return np.cos(ks * LENGTH / 2) ** 2


def configuration(mode, label, element):
    """Tracked R11, matrix-output R11 and the spin of an on-axis particle with spin (1, 0, 0)."""
    orbit, result = track(
        mode,
        element,
        P0,
        {"x": X0, "spz": 1.0},
        after_setup=MATRIX,
        outputs=("out.sdds", "matrix.sdds"),
    )
    matrix = result["matrix.sdds"]
    names = list(np.asarray(matrix[0, "ElementName"]))
    if names[-1] != "S":
        sys.exit(f"{label}: the matrix output does not end at the solenoid")
    printed = float(np.asarray(matrix[0, "R11"])[-1])
    # The symmetry (x, y) -> (-x, -y) of a solenoid excludes purely transverse second-order terms,
    # and the chromatic ones vanish on momentum, so x_out/x_in is R11 to rounding.
    tracked = orbit["x"] / X0
    spin, _ = track(mode, element, P0, {"spx": 1.0})
    return tracked, printed, np.array([spin["spx"], spin["spy"], spin["spz"]])


def check(mode, label, element, ks):
    """Compare tracking and matrix output with the strength ks; return per-quantity agreement."""
    tracked, printed, spin = configuration(mode, label, element)
    ok_track = compare(f"{mode} {label}: tracked R11", tracked, r11(ks), atol=ATOL)
    ok_matrix = compare(f"{mode} {label}: matrix-output R11", printed, r11(ks), atol=ATOL)
    ok_spin = compare(
        f"{mode} {label}: spin",
        spin,
        axial_rotation(on_axis_angle(ks, LENGTH, A_ELECTRON)),
        atol=ATOL,
    )
    return ok_track, ok_matrix, ok_spin


# The strength that B stands for: KS = q B/(m c p0) with the electron's physical charge -e.
KS_OF_B = -QE * FIELD / (ME * C * P0)
print(f"KS = {KS} /m; B = {FIELD} T stands for KS = {KS_OF_B:.16g} /m")
defect = False
for mode in modes():
    if not all(check(mode, "KS only (control)", f"SOLE,L={LENGTH!r},KS={KS!r}", KS)):
        sys.exit("The KS-only control is inconsistent: it is not valid.")
    if not all(check(mode, "B only (control)", f"SOLE,L={LENGTH!r},B={FIELD!r}", KS_OF_B)):
        sys.exit("The B-only control is inconsistent: it is not valid.")
    # Both given: the parameter description of B reads "field strength (used if KS is zero)", so
    # KS governs; the matrix output follows it, tracking does not.
    ok_track, ok_matrix, ok_spin = check(
        mode, "KS and B, expected per KS", f"SOLE,L={LENGTH!r},KS={KS!r},B={FIELD!r}", KS
    )
    if not ok_matrix:
        sys.exit("The matrix output with both strengths does not follow KS: unexpected behaviour.")
    defect |= not ok_track or not ok_spin
print(f"comparison: atol={ATOL:g}")
verdict(defect)
