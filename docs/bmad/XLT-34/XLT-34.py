"""XLT-34: SAD import reverses ROTATE on zero-length QUAD, SEXT and OCT

Importing a zero-length SAD QUAD, SEXT or OCT reverses the sense of its ROTATE angle. For a quadrupole with K1 = 0.5 m⁻¹ and ROTATE = 0.17 rad, a particle at (x, y) = (1, 2) mm receives normalized transverse momenta (pₓ, pᵧ) = (−0.8049, 0.7760) mrad instead of (−0.1379, 1.109) mrad. SAD’s rotation requires the negative angle in Bmad’s per-order tilt. Finite-length imports and equivalent skew MULT inputs use the correct convention, exposing the zero-length branch’s sign reversal.

The script runs the importer selected by BMAD_SOURCE on three thin magnet types and evaluates their kicks using SAD’s frame rotations and multipole formula. It compares corrected-angle, finite-length, and equivalent skew-MULT controls, then checks normal export/import round trips.
"""

import math

from _bmad import compare, verdict
from _s3_24_translations import lattice, sad_deck, sad_kick, translate_sad
from _s4_17_translations import export, track

bad = False
for typ, n, k in [("QUAD", 1, 0.5), ("SEXT", 2, 2.0), ("OCT", 3, 10.0)]:
    deck = sad_deck(f"{typ} M=(K{n}={k} ROTATE=.17);")
    thin = translate_sad(deck)
    expected = sad_kick(n, k, 0.17)
    native, _ = track(thin)
    bad |= compare(typ + " imported kick", native["FIRST"][[1, 3]], expected, atol=3e-16)
    corrected = lattice(f"multipole,k{n}l={k},t{n}=-.17")
    fixed, _ = track(corrected)
    if compare(typ + " negative-angle control", fixed["FIRST"][[1, 3]], expected, atol=3e-16):
        raise RuntimeError("Corrected-angle control failed")
    thick = translate_sad(sad_deck(f"{typ} M=(L=1e-6 K{n}={k} ROTATE=.17 DISFRIN=1);"))
    values, _ = track(thick)
    # The finite body differs by O(K*L); retain the measured residual explicitly.
    residual = values["FIRST"][[1, 3]] - expected
    print(f"{typ} L=1e-6 minus thin-map control: {residual.tolist()}")
    if abs(residual).max() > 1e-10:
        raise RuntimeError("Finite-length rotation control failed")
for typ, n, k, degrees in [("QUAD", 1, 0.2, 45.0), ("SEXT", 2, 2.0, 30.0), ("OCT", 3, 10.0, 22.5)]:
    rotated = translate_sad(sad_deck(f"{typ} M=(K{n}={k} ROTATE={degrees} DEG);"))
    skew = translate_sad(sad_deck(f"MULT M=(SK{n}={k});"))
    expected = sad_kick(n, k, math.radians(degrees))
    native, _ = track(rotated)
    control, _ = track(skew)
    bad |= compare(
        typ + " skew-equivalent rotated kick", native["FIRST"][[1, 3]], expected, atol=3e-16
    )
    if compare(typ + " MULT skew control", control["FIRST"][[1, 3]], expected, atol=3e-16):
        raise RuntimeError("Skew-MULT control failed")
for ele in [
    "multipole,k1l=.5,tilt=.17",
    "multipole,k2l=2,tilt=.17",
    "quadrupole,l=.5,k1=.3,tilt=.17",
]:
    lat = lattice(ele, "d:drift,l=1", "m,d")
    original, _ = track(lat)
    restored, _ = track(translate_sad(export(lat, "sad")))
    if compare(
        ele + " regular round-trip control",
        restored["ORBIT"][:4],
        original["ORBIT"][:4],
        atol=8e-15,
    ):
        raise RuntimeError("Regular SAD round-trip control failed")
verdict(bad)
