"""XLT-15: MAD-X initial dispersion omits the PT derivative conversion

Exporting initial dispersion at finite reference speed copies Bmad derivatives with respect to relative momentum into MAD-X fields defined with respect to PT, the normalized energy offset. At the reference orbit the chain rule requires division by β₀, the reference speed divided by light speed. For a 1 MeV/c electron in a 0.5 m drift, the imported horizontal dispersion becomes 0.7569 m instead of 0.8500 m, an error of 10.95%. Scaling the four exported initial dispersions by 1/β₀ restores the independent drift result.

The script exports zero starting coordinates with nonzero dispersion, checks Bmad drift transport, and loads the output into MAD-X with exactly matched mass and momentum. It compares the physical dispersion with the analytic drift result and repeats after changing only the four initial dispersion fields.
"""

import math
import re

import numpy as np
from _bmad import compare, verdict
from _lattice_import_export import HEADER, definition, export, inspect, madx, number, records, valid

mass, p0 = 510998.95069, 1e6
beta0 = p0 / math.hypot(p0, mass)
lattice = (
    HEADER.replace("3e9", "1e6")
    + """beginning[eta_x]=.8
beginning[etap_x]=.1
beginning[eta_y]=-.3
beginning[etap_y]=.04
e:drift,l=.5
lat:line=(e)
use,lat
"""
)
expected = np.array([0.85, 0.1, -0.28, 0.04])
result = inspect(lattice)
valid(result)
if compare(
    "Bmad drift dispersion control", records(result, "DISP")[-1], expected, atol=0, rtol=3e-15
):
    raise RuntimeError("Bmad drift control failed")
text = export(lattice)
initial = definition(text, "initial")
assert [number(initial, k) for k in ["x", "px", "y", "py", "t", "pt"]] == [0] * 6
keys = ["dx", "dpx", "dy", "dpy"]


def dispersion(text):
    with madx() as mad:
        mad.input(text)
        mad.input(
            f"beam,mass={mass * 1e-9:.17g},charge=-1,pc={p0 * 1e-9:.17g};use,period=lat;twiss,beta0=initial;"
        )
        np.testing.assert_allclose(mad.sequence.lat.beam.beta, beta0, rtol=2e-15)
        np.testing.assert_allclose(mad.table.twiss.s[-1], 0.5, rtol=2e-15)
        # dPT/ddelta=beta0 at the reference: convert MAD-X output back to d/ddelta.
        return beta0 * np.array([mad.table.twiss[k][-1] for k in keys])


actual = dispersion(text)
prefix, rest = text.split("initial: beta0,")
start, suffix = rest.split(";", 1)
for key, value in zip(keys, [0.8, 0.1, -0.3, 0.04], strict=True):
    start, count = re.subn(rf"\b{key}\s*=\s*[^,;\s]+", f"{key}={value / beta0:.17g}", start)
    assert count == 1
corrected = dispersion(prefix + "initial: beta0," + start + ";" + suffix)
if compare("MAD-X corrected derivative control", corrected, expected, atol=0, rtol=3e-15):
    raise RuntimeError("Corrected dispersion control failed")
verdict(compare("MAD-X delta-derivative dispersion", actual, expected, atol=0, rtol=3e-15))
