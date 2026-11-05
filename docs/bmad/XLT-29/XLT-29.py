"""XLT-29: MAD-X export loses order-zero multipole kicks at the reference momentum

Exporting an ab_multipole’s dipole components a₀ and b₀, or the corresponding end-multipole errors of a thick magnet, to MAD-X removes their kicks at the reference momentum. Bmad applies normalized momentum changes Δpₓ = −b₀ and Δpᵧ = a₀, but MAD-X interprets the exported order-zero coefficients as design terms and gives no on-momentum kick. For a₀ = 0.0007300 on an ab_multipole or a 0.5 m quadrupole with zero main strength, Bmad gives pᵧ = 0.0007300 while MAD-X TRACK and TWISS give zero. Handwritten MAD-X multipoles establish the same receiving-code behavior independently of the exporter.

The script exports with bmad_to_mad_sad_elegant a thin `ab_multipole, a0 = 0.73e-3` and a 0.5 m
quadrupole carrying the same a0 as a field error, and tracks a particle from a zero orbit
through each export with MAD-X TRACK and TWISS. The exit py is compared with Bmad's tracking
and with the order-zero kick Δpy = a0 on a straight reference. A hand-written MAD-X multipole
with ksl = {0.73e-3} or knl = {0.73e-3} shows that MAD-X gives no order-zero kick in TRACK,
TWISS or PTC.
"""

from _bmad import compare, verdict
from _s4_17_translations import export, tao
from _translators import lattice, madx, madx_track, numbers, statements

A0 = 7.3e-4
# Bmad's order-zero terms are plain dipole kicks on a straight reference: Delta px = -b0,
# Delta py = +a0. Thick elements apply half of each multipole kick at either end.
CASES = {
    "ab_multipole": "e: ab_multipole, a0 = 7.3e-4",
    "quadrupole with a0 error": "e: quadrupole, l = 0.5, a0 = 7.3e-4, scale_multipoles = F",
}


def ptc_end(mad):
    mad.input(
        "ptc_create_universe; ptc_create_layout, model = 2, method = 6, nst = 10, exact = true;"
        "ptc_start, x = 0, px = 0, y = 0, py = 0, t = 0, pt = 0;"
        "ptc_track, icase = 6, element_by_element = false, onetable = true;"
        "ptc_track_end; ptc_end;"
    )
    return mad.table.trackone


defect = False
for label, element in CASES.items():
    lat = lattice(element, "e")
    bmad = numbers(tao(lat, ["pipe ele:orbit 1"])["pipe ele:orbit 1"], ["px", "py"])
    if compare(f"{label}: Bmad exit (px, py), expected (0, a0)", bmad, [0, A0], atol=1e-18):
        raise SystemExit("Bmad disagrees with the order-zero kick: comparison not valid.")
    deck = export(lat, "madx")
    for name, text in statements(deck, "madx").items():
        if "multipole" in text:
            print(f"{label}: exported: {name}: {text}")
    with madx(deck) as mad:
        twiss = mad.twiss(beta0="initial")
        twiss_end = [twiss.px[-1], twiss.py[-1]]
        track_end = madx_track(mad, [0, 0, 0, 0, 0, 0])[[1, 3]]
    defect |= compare(f"{label}: MAD-X TRACK exit (px, py)", track_end, [0, A0], atol=1e-18)
    defect |= compare(f"{label}: MAD-X TWISS exit (px, py)", twiss_end, [0, A0], atol=1e-18)

# MAD-X itself: a multipole's order-zero coefficient bends the reference orbit (manual) and
# gives no kick to an on-momentum particle in these three modules.
for array in ("ksl", "knl"):
    deck = (
        "beam, particle = electron, pc = 1;\n"
        f"e: multipole, {array} = {{{A0}}};\nlat: line = (e);\nuse, period = lat;\n"
    )
    with madx(deck) as mad:
        twiss = mad.twiss(betx=10, bety=10)
        track = madx_track(mad, [0, 0, 0, 0, 0, 0])
        table = ptc_end(mad)
        values = [twiss.px[-1], twiss.py[-1], track[1], track[3], table.px[-1], table.py[-1]]
        values = [float(v) for v in values]
    print(f"hand-written {array} = {{{A0}}}: TWISS, TRACK, PTC exit (px, py) = {values}")
    if any(values):
        raise SystemExit("MAD-X gives an order-zero kick: the receiving-code premise fails.")
verdict(defect)
