"""XLT-13: MAD-X skew-multipole export reverses the kick

Exporting a skew multipole of quadrupole order or higher to MAD-X negated its skew coefficients
and reversed its transverse kick. The shared complex field expansion requires the same skew sign
in both codes. For integrated skew-quadrupole strength 0.91 m⁻¹ at (x, y) = (1, 2) mm, the
correct normalized transverse momentum changes are (0.001820, 0.0009100); the export gave
(−0.001820, −0.0009100). The field expansion and Bmad tracking agree on the positive changes.
The same writer handles skew field errors attached to thick magnets, while the main fields of
thick skew magnets and the MAD8 export use separate, correct branches.

The script exports with bmad_to_mad_sad_elegant a thin skew quadrupole given as
`ab_multipole, a1 = 0.91` and as `multipole, k1sl = 0.91`, reads the written `ksl`, and tracks
a particle at (x, y) = (1, 2) mm through each export with MAD-X TRACK. The kick is compared
with the field expansion (b₁ + i a₁)(x + i y) that Bmad and MAD-X share, and Bmad's own kick is
the control. A copy of the export with `ksl` set to +0.91 is the receiving-code control.
"""

import re

import numpy as np
from _bmad import compare, verdict
from _translators import export, lattice, madx, madx_track, numbers, statements, tao

X, Y, STRENGTH = 1e-3, 2e-3, 0.91
# Thin skew quadrupole: F = (b1 + i a1)(x + i y) with b1 = 0, a1 = 0.91;
# Delta px = -Re F, Delta py = +Im F.
field = 1j * STRENGTH * complex(X, Y)
EXPECTED = np.array([-field.real, field.imag])

defect = False
for element in ("ab_multipole, a1 = 0.91", "multipole, k1sl = 0.91"):
    lat = lattice(f"e: {element}", "e", start=[X, 0, Y, 0, 0, 0])
    kick = numbers(tao(lat, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["px", "py"])
    if compare(f"{element}: Bmad (px, py)", kick, EXPECTED, atol=0, rtol=2e-15):
        raise SystemExit("Bmad disagrees with the field expansion: comparison not valid.")
    deck = export(lat, "madx")
    print(f"{element}: exported: E: {statements(deck, 'madx')['E']}")
    with madx(deck) as mad:
        end = madx_track(mad, [X, 0, Y, 0, 0, 0])
    defect |= compare(
        f"{element}: MAD-X TRACK (px, py) of the export", end[[1, 3]], EXPECTED, atol=0, rtol=2e-15
    )
    # Receiving-code controls: ksl = +0.91 must give the field-expansion kick, and the value
    # -0.91 that release 20261020-0 wrote must give the reversed kick.
    for sign, target in ((1, EXPECTED), (-1, -EXPECTED)):
        edited, count = re.subn(r"ksl = \{[^}]+\}", f"ksl = {{0, {sign * STRENGTH}}}", deck)
        if count != 1:
            raise SystemExit("Expected one ksl array in the export.")
        with madx(edited) as mad:
            end = madx_track(mad, [X, 0, Y, 0, 0, 0])
        if compare(
            f"{element}: MAD-X TRACK with ksl = {{0, {sign * STRENGTH}}}, expected {sign:+d} x field",
            end[[1, 3]],
            target,
            atol=0,
            rtol=2e-15,
        ):
            raise SystemExit("A ksl control disagrees: comparison not valid.")
verdict(defect)
