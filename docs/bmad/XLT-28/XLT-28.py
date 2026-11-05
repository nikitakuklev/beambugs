"""XLT-28: MAD8 export of a zero-length lcavity drops its energy gain

Exporting a zero-length traveling-wave lcavity to MAD8 loses its specified accelerating voltage. A 1 MV cavity at crest is written without DELTAE, the MAD8 energy-gain parameter, giving zero gain instead of 1 MeV; the same voltage at length 0.5 m writes DELTAE = 1 MeV. The writer computes gradient times length, which vanishes for the valid zero-length representation. The stored voltage and the finite-length control establish the missing energy gain.

The script exports zero-length and finite-length traveling-wave cavities and reads their MAD8 energy-gain parameters. It also loads the Bmad lattices to check their stored voltage and gradient, using the finite-length case as a control.
"""

import re

from _bmad import compare, verdict
from _s3_24_translations import lattice
from _s4_17_translations import export, track

bad = False
for length in (0, 0.5):
    lat = lattice(f"lcavity,l={length},voltage=1e6,rf_frequency=5e8,cavity_type=traveling_wave")
    native, _ = track(lat, start=(0, 0, 0, 0, 0, 0))
    if compare(f"L={length} Bmad voltage control (V)", native["ATTR"][1], 1e6, atol=1e-9):
        raise RuntimeError("Input voltage was not retained")
    text = export(lat, "mad8")
    line = next(s for s in text.split(";") if re.search(r"(?m)^M:", s))
    match = re.search(r"deltae\s*=\s*([\d.Ee+-]+)", line, re.IGNORECASE)
    # MAD8's absent DELTAE has its zero default.
    actual = float(match[1]) if match else 0.0
    print(
        f"L={length}: stored gradient={native['ATTR'][2]:.16g} V/m; DELTAE written={match is not None}"
    )
    mismatch = compare(f"L={length} MAD8 DELTAE (MeV), voltage/1e6", actual, 1.0, atol=1e-14)
    if length and mismatch:
        raise RuntimeError("Finite-length control failed")
    bad |= mismatch
verdict(bad)
