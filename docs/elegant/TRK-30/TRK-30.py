"""TRK-30: Matrix concatenation drops the spin rotation of SOLE and ROTATE

With spin tracking enabled and a nonzero concat_order in run_setup, Elegant replaces each sequence
of matrix elements of at most that order, even a single element, by one generic matrix element that
transports only the orbit, so the spin rotations of absorbed SOLE solenoids and ROTATE elements are
lost. For an on-axis electron at momentum 200 mₑc in a 0.65 m SOLE with KS = 0.9000 m⁻¹, the
Thomas–BMT precession of 0.5857 rad, which concat_order = 0 reproduces, vanishes at concat_order =
2 and 3 while the orbit is unchanged. A ROTATE with TILT = 0.3000 rad leaves the spin unrotated at
concat_order = 1. The only warning says that concatenation reduces accuracy.

The script tracks an on-axis electron with spin (1, 0, 0) through one order-2 SOLE at concat_order
0 to 3 and compares the exit spin with the exact on-axis Thomas–BMT precession; concat_order 0 and
1, which leave the element in place, are the controls. It checks that an off-axis orbit is
identical with and without concatenation, and repeats the comparison for a ROTATE element against
its own unconcatenated frame rotation.
"""

import sys

import numpy as np
from _elegant import modes, verdict
from _sole import A_ELECTRON, axial_rotation, compare, on_axis_angle, track

LENGTH, P0, KS = 0.65, 200.0, 0.9
TILT = 0.3  # rad
ATOL = 1e-13


def spin_through(mode, element, concat_order, start=None):
    """Final spin and orbit of one particle through `element` with the given concat_order."""
    out, result = track(
        mode, element, P0, start or {"spx": 1.0}, setup=f", concat_order={concat_order}"
    )
    if concat_order and "concat_order is non-zero" not in result["log"]:
        sys.exit("The concatenation warning is missing: the run did not concatenate as assumed.")
    return np.array([out["spx"], out["spy"], out["spz"]]), out


defect = False
sole = f"SOLE,L={LENGTH!r},KS={KS!r},ORDER=2"
# Thomas-BMT on the axis of the solenoid (see _sole.on_axis_angle).
expected_sole = axial_rotation(on_axis_angle(KS, LENGTH, A_ELECTRON))
# ROTATE turns the coordinate frame by TILT about +z, so the components of a fixed spin turn
# by -TILT.
expected_rotate = axial_rotation(-TILT)
for mode in modes():
    # concat_order = 0 (no concatenation) and 1 (the order-2 SOLE matrix is not absorbed) keep SOLE.
    for order in (0, 1):
        spin, _ = spin_through(mode, sole, order)
        if not compare(
            f"{mode} SOLE, concat_order={order} (control): spin", spin, expected_sole, atol=ATOL
        ):
            sys.exit("The unconcatenated SOLE control disagrees with Thomas-BMT: it is not valid.")
    for order in (2, 3):
        spin, _ = spin_through(mode, sole, order)
        defect |= not compare(
            f"{mode} SOLE, concat_order={order}: spin", spin, expected_sole, atol=ATOL
        )
    # The orbit is not affected: an off-axis particle leaves concat_order 0 and 2 identically.
    start = {"x": 1e-3, "xp": -2e-4, "y": 5e-4, "yp": 3e-4, "spz": 1.0}
    _, plain = spin_through(mode, sole, 0, start)
    _, merged = spin_through(mode, sole, 2, start)
    names = ("x", "xp", "y", "yp", "t", "p")
    diff = max(abs(plain[n] - merged[n]) for n in names)
    print(f"{mode} SOLE orbit, concat_order 2 minus 0: max|difference|={diff:.3g}")
    if diff != 0:
        sys.exit("Concatenation changed the orbit: the comparison is not isolated to the spin.")
    spin, _ = spin_through(mode, f"ROTATE,TILT={TILT!r}", 0)
    if not compare(
        f"{mode} ROTATE, concat_order=0 (control): spin", spin, expected_rotate, atol=ATOL
    ):
        sys.exit("The ROTATE control disagrees with the frame rotation: it is not valid.")
    spin, _ = spin_through(mode, f"ROTATE,TILT={TILT!r}", 1)
    defect |= not compare(f"{mode} ROTATE, concat_order=1: spin", spin, expected_rotate, atol=ATOL)
print(f"comparison: atol={ATOL:g}")
verdict(defect)
