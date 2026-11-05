"""XLT-22: MAD-X export of a bend vkick is lost outside PTC

Exporting a bend with a vertical steering kick to MAD-X writes the kick into k0s, a skew dipole strength used by PTC but ignored by ordinary TWISS and TRACK. For a 1.5 m bend with angle 0.15 rad and vkick = 1 mrad, Bmad and MAD-X PTC give vertical displacement y = 0.7500 mm and normalized vertical momentum pᵧ = 0.001000. This agrees with a kick distributed along the bend, whereas ordinary MAD-X TWISS and TRACK give zero displacement and momentum. The exported lattice therefore loses this steering in those two calculation modes.

The script exports with bmad_to_mad_sad_elegant an sbend with L = 1.5 m, angle 0.15 rad and
vkick = 1 mrad, and propagates a particle from a zero orbit through the export with MAD-X
TWISS, MAD-X TRACK and MAD-X PTC tracking. Each exit (y, py) is compared with the uniform
vertical kick, y = vkick L/2 and py = vkick. Bmad's own tracking and the PTC result, which reads
the exported k0s, are the controls.
"""

from _bmad import compare, verdict
from _s4_17_translations import export, tao
from _translators import lattice, madx, madx_track, numbers, statements

LENGTH, ANGLE, VKICK = 1.5, 0.15, 1e-3
# A vertical kick spread uniformly over L: y = vkick L/2, py = vkick to first order.
# The leading kinematic correction in y is L*vkick**3/8 = 1.875e-10 m.
# A 1e-9 absolute tolerance resolves the missing kick while allowing these cubic terms.
EXPECTED = [VKICK * LENGTH / 2, VKICK]
ATOL = 1e-9

lat = lattice(f"b: sbend, l = {LENGTH}, angle = {ANGLE}, vkick = {VKICK}", "b")
bmad = numbers(tao(lat, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["y", "py"])
if compare("Bmad exit (y m, py), expected uniform kick", bmad, EXPECTED, atol=ATOL):
    raise SystemExit("Bmad disagrees with the uniform kick: comparison not valid.")

deck = export(lat, "madx")
print(f"exported: B: {statements(deck, 'madx')['B']}")
with madx(deck) as mad:
    twiss = mad.twiss(beta0="initial")
    twiss_end = [twiss.y[-1], twiss.py[-1]]
    track_end = madx_track(mad, [0, 0, 0, 0, 0, 0])[[2, 3]]
    mad.input(
        "ptc_create_universe; ptc_create_layout, model = 2, method = 6, nst = 10, exact = true;"
        "ptc_start, x = 0, px = 0, y = 0, py = 0, t = 0, pt = 0;"
        "ptc_track, icase = 6, element_by_element = false, onetable = true;"
        "ptc_track_end; ptc_end;"
    )
    table = mad.table.trackone
    ptc_end = [table.y[-1], table.py[-1]]
if compare("MAD-X PTC exit (y m, py) of the export", ptc_end, EXPECTED, atol=ATOL):
    raise SystemExit("PTC disagrees with the uniform kick: the exported k0s is not the kick.")
defect = compare("MAD-X TWISS exit (y m, py) of the export", twiss_end, EXPECTED, atol=ATOL)
defect |= compare("MAD-X TRACK exit (y m, py) of the export", track_end, EXPECTED, atol=ATOL)
verdict(defect)
