"""XLT-07: MAD-X bend import treats total k0 as a field error

Importing a MAD-X sector bend with explicit nonzero total dipole strength k₀ assigns Bmad's
field error dg = k₀, adding the design curvature g = angle/L to the intended field. For an ideal
1 m bend with angle 0.1 rad and k₀ = 0.1 m⁻¹, this doubles the field to 0.2 m⁻¹. An on-axis
particle then exits at x = −49.96 mm and normalized horizontal momentum pₓ = −0.09983, matching
an independent circle calculation, while MAD-X keeps it on the design orbit. The correct import
without taper is dg = k₀ − g, zero here. The same sector bend without explicit k₀ imports
correctly.

The script translates with madx_to_bmad.py a MAD-X sbend with L = 1 m, angle 0.1 rad and k0 = 0.1
m⁻¹, the ideal bend written with its field explicit, and the same bend without k0 as the control. It
reads the imported dg and tracks the imported lattice in Tao from a zero orbit. MAD-X TWISS of the
source deck, which must stay on the design orbit, is the receiving-code reference, and the exact
circle in the doubled field accounts for the observed orbit.
"""

from _bmad import compare, verdict
from _translators import bend_exit, import_madx, madx, numbers, tao, value

G, LENGTH = 0.1, 1.0
SOURCE = (
    "beam, particle = electron, energy = 3;\n"
    f"b: sbend, l = {LENGTH}, angle = {G * LENGTH}{{}};\n"
    "ll: line = (b);\nuse, period = ll;\n"
)
OPEN = "parameter[geometry] = open\nbeginning[beta_a] = 10\nbeginning[beta_b] = 10\n"


def imported_orbit(deck):
    """Imported Bmad text and the Tao exit (x, px) of the imported bend from a zero orbit."""
    text = import_madx(deck)
    orbit = numbers(tao(text + OPEN, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["x", "px"])
    return text, orbit


defect = False
for k0 in (None, G):
    deck = SOURCE.format("" if k0 is None else f", k0 = {k0}")
    label = "no k0" if k0 is None else f"k0 = {k0}"
    with madx(deck) as mad:
        twiss = mad.twiss(betx=10, bety=10)
        madx_end = [twiss.x[-1], twiss.px[-1]]
    # k0 = h = angle/L is MAD-X's ideal bend: the orbit stays on the design circle.
    if compare(
        f"{label}: MAD-X TWISS exit (x m, px), expected design orbit", madx_end, [0, 0], atol=1e-15
    ):
        raise SystemExit("MAD-X leaves the design orbit: comparison not valid.")
    text, orbit = imported_orbit(deck)
    bend = next(s for s in text.splitlines() if s.startswith("b:"))
    print(f"{label}: imported bend: {bend}")
    observed_dg = value(bend, "dg", 0.0)
    bad = compare(f"{label}: imported dg (1/m), expected k0 - angle/L", observed_dg, 0, atol=1e-15)
    bad |= compare(
        f"{label}: Bmad exit (x m, px), expected design orbit", orbit, [0, 0], atol=1e-14
    )
    if k0 is None and bad:
        raise SystemExit("The bend without k0 is not imported as an ideal bend: not valid.")
    if k0 is not None:
        defect |= bad
        # The orbit is that of the exact circle in g + dg = 0.2 1/m (closed form, rounding 1e-15 m).
        doubled = bend_exit(G, G + observed_dg, G * LENGTH)
        if compare(
            f"{label}: Bmad exit, expected exact circle in g + dg", orbit, doubled, atol=1e-14
        ):
            raise SystemExit("The imported orbit is not the circle of g + dg: not valid.")
verdict(defect)
